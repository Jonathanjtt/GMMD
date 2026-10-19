"""Stage B: the conditional mixture network ``q_phi(y | x, j)`` and its likelihood.

Architecture.  The teacher's own NCSN++ (Song et al., CelebA-HQ 256 config, 65.6 M parameters),
initialised from the checkpoint's EMA weights, with two changes:

* the seven output-skip convolutions (``conv3x3(in_ch, 3)``, one per resolution) are widened to
  ``3 K`` channels by tiling the teacher's kernels K times plus a small seeded perturbation that
  breaks the symmetry between components.  The network therefore returns K score-like fields
  ``h_k(x, sigma_j)`` from ONE forward pass;
* a global head reads the bottleneck (4 x 4) features, pooled, plus a learned interval embedding,
  and returns K mixture logits and K log-variance offsets (zero-initialised last layer).

Mixture, per interval ``j`` (start state ``x`` at ``sigma_j``):

    q_phi(y | x, j) = sum_k pi_k N(y; mu_k, v_k I)
    pi = softmax(logits)                     global weights over the whole state
    mu_k = x + g_j h_k                       g_j = sigma_j^2 - sigma_end^2 (sigma_j^2 at the end)
    log v_k = log v_ref,j + 6 tanh(delta_k / 6)

At initialisation every mu_k is the teacher's first-order Tweedie mean of the end state,
``x + g_j s_theta(x, t_j)``, and v_k = v_ref,j: the student starts as (K copies of) the coarse
ancestral sampler and has to learn everything IBW adds.  The likelihood sums each component's
log density over all d coordinates BEFORE the log-sum-exp over components, so a sample is never
assembled from different components in different pixels.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MixtureStudent", "output_convs", "build_ncsnpp", "mixture_log_prob",
           "standardized_nll", "count_parameters"]

_LOG2PI = math.log(2 * math.pi)
LOGV_BOUND = 6.0


def build_ncsnpp(config_name):
    """An NCSN++ of the named official config with random weights (no checkpoint involved)."""
    from gmmd.teachers.score_sde._vendor.models import ncsnpp  # noqa: F401  (registers 'ncsnpp')
    from gmmd.teachers.score_sde._vendor.models import utils as mutils
    from gmmd.teachers.score_sde.configs import get_config

    cfg = get_config(config_name)
    return mutils.get_model(cfg.model.name)(cfg), cfg


def output_convs(net, channels):
    """Indices in ``net.all_modules`` of the output-skip convolutions (out_channels == channels)."""
    idx = [i for i, m in enumerate(net.all_modules)
           if isinstance(m, nn.Conv2d) and m.out_channels == channels]
    if net.config.model.progressive != "output_skip" or len(idx) != len(net.config.model.ch_mult):
        raise NotImplementedError("the student widens NCSN++'s output-skip pyramid; this config "
                                  f"has progressive={net.config.model.progressive!r}, {len(idx)} "
                                  "candidate output convolutions")
    return idx


def _bottleneck_index(net):
    """Index of the last middle ResnetBlock (the 4 x 4 bottleneck), computed as the constructor
    lays the modules out, then type-checked."""
    from gmmd.teachers.score_sde._vendor.models import layerspp

    c = net.config.model
    res = [net.config.data.image_size // 2 ** i for i in range(len(c.ch_mult))]
    i = 4                                    # Fourier proj, 2 x Linear, input conv
    for lvl in range(len(c.ch_mult)):
        i += c.num_res_blocks * (2 if res[lvl] in c.attn_resolutions else 1)
        if lvl != len(c.ch_mult) - 1:
            i += 2                           # down ResnetBlock + input-skip Combine
    i += 2                                   # middle: ResnetBlock, AttnBlock, -> ResnetBlock at i
    mods = net.all_modules
    if not (isinstance(mods[i], layerspp.ResnetBlockBigGANpp)
            and isinstance(mods[i - 1], layerspp.AttnBlockpp)):
        raise RuntimeError("could not locate the NCSN++ bottleneck block")
    return i


def count_parameters(module, trainable_only=False):
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only))


class MixtureStudent(nn.Module):
    """``(x, j) -> (logits (B, K), means (B, K, C, H, W), log_variances (B, K))``.

    ``intervals``: list of dicts with ``sigma_start``, ``mean_gap``, ``log_vref`` per interval.
    Build with :meth:`from_teacher` (training) or :meth:`from_checkpoint` (sampling; no teacher).
    """

    def __init__(self, config_name, intervals, n_components=8, head_width=512, embed_dim=64):
        super().__init__()
        net, cfg = build_ncsnpp(config_name)
        self.config_name = config_name
        self.K = K = int(n_components)
        self.C = C = int(cfg.data.num_channels)
        self.image_shape = (C, cfg.data.image_size, cfg.data.image_size)
        self.net = net
        self._out_idx = output_convs(net, C)
        for i in self._out_idx:                                  # widen: (in, 3) -> (in, 3K)
            old = net.all_modules[i]
            new = nn.Conv2d(old.in_channels, C * K, kernel_size=3, padding=1, bias=True)
            net.all_modules[i] = new
        self._mid = _bottleneck_index(net)
        self._feat = None
        net.all_modules[self._mid].register_forward_hook(self._grab)
        ch = cfg.model.nf * cfg.model.ch_mult[-1]                # bottleneck channels
        self.n_intervals = len(intervals)
        self.embed = nn.Embedding(self.n_intervals, embed_dim)
        self.head = nn.Sequential(nn.LayerNorm(ch + embed_dim), nn.Linear(ch + embed_dim, head_width),
                                  nn.SiLU(), nn.Linear(head_width, head_width), nn.SiLU(),
                                  nn.Linear(head_width, 2 * K))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("sigma_start", torch.tensor([iv["sigma_start"] for iv in intervals],
                                                         dtype=torch.float32))
        self.register_buffer("mean_gap", torch.tensor([iv["mean_gap"] for iv in intervals],
                                                      dtype=torch.float32))
        self.register_buffer("log_vref", torch.tensor([iv["log_vref"] for iv in intervals],
                                                      dtype=torch.float32))
        self.intervals = [dict(iv) for iv in intervals]
        self.n_forward = 0              # images passed through forward() (instrumentation)
        self.n_calls = 0                # forward() calls

    def _grab(self, module, inputs, output):
        self._feat = output

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_teacher(cls, teacher_model, config_name, intervals, n_components=8, perturb=1e-2,
                     seed=0, **kw):
        """Copy the teacher's EMA weights in; tile the output kernels K times (+ perturbation)."""
        self = cls(config_name, intervals, n_components=n_components, **kw)
        sd = {k: v.detach().cpu().clone() for k, v in teacher_model.state_dict().items()}
        K, C = self.K, self.C
        g = torch.Generator().manual_seed(int(seed))
        for i in self._out_idx:
            w, b = sd[f"all_modules.{i}.weight"], sd[f"all_modules.{i}.bias"]
            wt, bt = w.repeat(K, 1, 1, 1).clone(), b.repeat(K).clone()
            noise = torch.randn(wt.shape, generator=g) * (float(perturb) * w.std())
            noise[:C] = 0.0                           # component 0 is the teacher exactly
            sd[f"all_modules.{i}.weight"], sd[f"all_modules.{i}.bias"] = wt + noise, bt
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        if unexpected or set(missing) - {"sigmas"}:
            raise RuntimeError(f"teacher->student weight copy: missing {missing}, unexpected {unexpected}")
        self.net.all_modules[0].W.requires_grad_(False)        # Fourier features stay fixed
        return self

    def arch(self):
        return dict(config_name=self.config_name, n_components=self.K, intervals=self.intervals,
                    head_width=self.head[1].out_features, embed_dim=self.embed.embedding_dim)

    @classmethod
    def from_arch(cls, arch):
        return cls(arch["config_name"], arch["intervals"], n_components=arch["n_components"],
                   head_width=arch["head_width"], embed_dim=arch["embed_dim"])

    # ---------------------------------------------------------------- forward
    def forward(self, x, j):
        B = x.shape[0]
        j = torch.as_tensor(j, device=x.device).long().reshape(-1)
        if j.numel() == 1:
            j = j.expand(B)
        self.n_calls += 1
        self.n_forward += B
        h = self.net(x, self.sigma_start[j])                                   # (B, 3K, H, W)
        feat = self._feat.float().mean((2, 3))
        self._feat = None
        out = self.head(torch.cat([feat, self.embed(j)], 1)).float()
        logits, dlogv = out[:, :self.K], out[:, self.K:]
        means = x[:, None] + self.mean_gap[j].view(B, 1, 1, 1, 1) * \
            h.float().view(B, self.K, *self.image_shape)
        logv = self.log_vref[j][:, None] + LOGV_BOUND * torch.tanh(dlogv / LOGV_BOUND)
        return logits, means, logv


# ======================================================================================
# likelihood
# ======================================================================================
def mixture_log_prob(z, means, logv, logits):
    """``log q(z)`` for a batch of isotropic mixtures with global weights.

    ``z`` (B, M, d); ``means`` (B, K, d); ``logv`` (B, K); ``logits`` (B, K) -> (B, M).
    Each component's Gaussian log density is summed over the d coordinates first, then mixed."""
    B, M, d = z.shape
    K = means.shape[1]
    quad = torch.stack([((z - means[:, k:k + 1]) ** 2).sum(-1) for k in range(K)], -1)   # (B,M,K)
    logn = -0.5 * (d * (_LOG2PI + logv)[:, None, :] + quad / logv.exp()[:, None, :])
    return torch.logsumexp(F.log_softmax(logits, -1)[:, None, :] + logn, -1)


def standardized_nll(z, x, means, logv, logits, log_scale):
    """Per-dimension ``-log q(z)`` in standardised coordinates ``u = (z - x) / scale``.

    The change of variables shifts the NLL by the constant ``d log scale`` per sample, so the
    minimiser is unchanged while every interval contributes gradients of comparable size.
    ``z`` (B, M, d), ``x`` (B, d), ``means`` (B, K, d), ``log_scale`` (B,) -> (B, M)."""
    d = z.shape[-1]
    sc = log_scale.exp()[:, None, None]
    u = (z - x[:, None]) / sc
    mu = (means - x[:, None]) / sc
    lv = logv - 2 * log_scale[:, None]
    return -mixture_log_prob(u, mu, lv, logits) / d
