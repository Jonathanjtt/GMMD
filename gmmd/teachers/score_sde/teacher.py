"""Song et al.'s score SDE as the GMMD *stochastic teacher*.

The teacher answers one question: given a FIXED noisy image ``x_t`` at diffusion time ``t``,
draw ``x_s ~ p_theta(x_s | x_t)`` for ``s < t`` by running the official reverse-time sampler
from ``t`` down to ``s`` -- and nothing else.  Everything numerical is the vendored upstream
code (:mod:`._vendor.sde_lib`, :mod:`._vendor.sampling`, the NCSN++ network); this module only
(a) loads the checkpoint the way ``run_lib.evaluate`` does (EMA weights copied into the model),
(b) restricts the official predictor-corrector loop to the index range ``[i_t, i_s)`` of the
official time grid ``linspace(T, eps, N)``, and (c) threads per-rollout random seeds through the
sampler's noise hook so that every rollout is independently reproducible.

Three samplers are exposed, all on the same grid:

* ``predictor="reverse_diffusion", corrector="none"`` -- the exact SMLD discretisation of the
  reverse SDE ``dx = -g(t)^2 score dt + g(t) dw`` (``sde_lib.VESDE.discretize`` +
  ``ReverseDiffusionPredictor``); ``predictor="euler_maruyama"`` is the Euler-Maruyama one;
* ``corrector="langevin"`` adds the official Langevin corrector (the celebahq_256 config's
  default PC sampler, ``snr = 0.075``, one corrector step per level).  NOTE: the corrector runs
  Langevin steps targeting the MARGINAL ``p_{t_i}`` at every level, so the transition kernel it
  induces is not the reverse SDE's own; which one is "the teacher" is a modelling choice --
  see the experiment config;
* ``probability_flow=True`` with ``corrector="none"`` -- the probability-flow ODE on the same
  grid (``rev_G = 0``, half the score term), the deterministic control.

``x_t`` is never modified: it is copied once and replicated across the rollout batch.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from ._vendor import sampling, sde_lib
from ._vendor.models import ncsnpp  # noqa: F401 -- import registers 'ncsnpp' in the model registry
from ._vendor.models import utils as mutils
from ._vendor.models.ema import ExponentialMovingAverage
from ._vendor.provenance import UPSTREAM_COMMIT
from .configs import SAMPLING_EPS, get_config, to_dict

__all__ = ["ScoreSDETeacher", "Transition", "build_sde", "load_score_model", "file_sha256"]

PREDICTORS = ("reverse_diffusion", "euler_maruyama", "ancestral_sampling")
CORRECTORS = ("none", "langevin", "ald")


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_sde(config):
    """The forward SDE of a config and the sampling ``eps`` -- exactly ``run_lib.py``'s branch."""
    kind = config.training.sde.lower()
    m = config.model
    if kind == "vesde":
        sde = sde_lib.VESDE(sigma_min=m.sigma_min, sigma_max=m.sigma_max, N=m.num_scales)
    elif kind == "vpsde":
        sde = sde_lib.VPSDE(beta_min=m.beta_min, beta_max=m.beta_max, N=m.num_scales)
    elif kind == "subvpsde":
        sde = sde_lib.subVPSDE(beta_min=m.beta_min, beta_max=m.beta_max, N=m.num_scales)
    else:
        raise ValueError(f"SDE {config.training.sde} unknown.")
    return sde, SAMPLING_EPS[kind]


def load_score_model(config, checkpoint, device="cuda", use_ema=True):
    """Build the network of ``config`` and load ``checkpoint`` into it.

    Mirrors ``utils.restore_checkpoint`` + ``run_lib.evaluate``: the saved ``model`` state dict
    was written from a ``torch.nn.DataParallel`` wrapper (hence the ``module.`` prefix, stripped
    here), every learned tensor must match the architecture the config builds (an
    architecture/config mismatch fails loudly rather than silently), and the EMA shadow
    parameters are copied into the model before sampling, which is what upstream's evaluation
    does (``ema.copy_to(score_model.parameters())``).
    """
    checkpoint = Path(checkpoint)
    model = mutils.get_model(config.model.name)(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in state["model"].items()}
    # Upstream loads with strict=False because the checkpoint never contains ``sigmas``, a buffer
    # NCSN++ derives from the config at construction.  Allow exactly that key and nothing else.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected or set(missing) - {"sigmas"}:
        raise RuntimeError(f"checkpoint/config mismatch: missing {sorted(missing)}, "
                           f"unexpected {sorted(unexpected)}")
    info = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                step=int(state["step"]), use_ema=bool(use_ema),
                n_params=int(sum(p.numel() for p in model.parameters())))
    if use_ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        ema.load_state_dict(state["ema"])
        ema.copy_to(model.parameters())
        info["ema_num_updates"] = int(ema.num_updates)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, info


@dataclass
class Transition:
    """One batch of rollouts from one fixed ``x_t`` to time ``s``."""
    x_s: torch.Tensor                      # (B, C, H, W), teacher device, the STOCHASTIC state at s
    x_s_mean: torch.Tensor                 # (B, C, H, W), the last predictor step's noise-free mean
    t: float                               # realised grid time of x_t
    s: float                               # realised grid time of x_s
    i_t: int
    i_s: int
    n_steps: int                           # predictor steps taken: i_s - i_t
    nfe_per_sample: int                    # score-network evaluations each rollout went through
    n_forward_calls: int                   # network forward passes (batches) this call made
    seeds: tuple
    method: dict
    runtime_s: float
    intermediates: Optional[np.ndarray] = None        # (n_snap, B, C, H, W) float32, CPU
    intermediate_steps: Optional[np.ndarray] = None   # grid index after each snapshot
    intermediate_times: Optional[np.ndarray] = None

    @property
    def x_s_np(self):
        return self.x_s.detach().cpu().numpy()


class ScoreSDETeacher:
    """A pretrained score SDE model wrapped as a fixed-``x_t`` transition sampler.

    Build with :meth:`from_config` (real checkpoint) or directly with a ``score_fn`` (tests: an
    analytic score, no network).  ``score_fn(x, t)`` takes ``x`` (B, C, H, W) and ``t`` (B,) and
    returns ``grad_x log p_t(x)`` in the same shape -- the convention of
    ``models.utils.get_score_fn``.
    """

    def __init__(self, sde, score_fn, image_shape, device="cpu", eps=None, *, config=None,
                 model=None, info=None, deterministic_cudnn=True):
        self.sde = sde
        self._score_fn = score_fn
        self.image_shape = tuple(int(v) for v in image_shape)
        self.device = torch.device(device)
        self.eps = float(SAMPLING_EPS.get(type(sde).__name__.lower(), 1e-3) if eps is None else eps)
        self.config = config
        self.model = model
        self.info = dict(info or {})
        self.n_score_calls = 0
        if deterministic_cudnn and self.device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Upstream keeps the discrete sigma / beta tables on the CPU and, in VESDE.discretize,
        # indexes ``discrete_sigmas[timestep - 1]`` with a CUDA index BEFORE moving it (modern
        # torch refuses); moving the table to the device once is the same arithmetic.
        for name in ("discrete_sigmas", "discrete_betas", "alphas", "alphas_cumprod",
                     "sqrt_alphas_cumprod", "sqrt_1m_alphas_cumprod"):
            if hasattr(self.sde, name):
                setattr(self.sde, name, getattr(self.sde, name).to(self.device))
        # the official sampling grid, built exactly as get_pc_sampler builds it
        self.timesteps = torch.linspace(self.sde.T, self.eps, self.sde.N, device=self.device)
        self._grid = self.timesteps.detach().cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config_name, checkpoint, device="cuda", use_ema=True, allow_tf32=False,
                    deterministic_cudnn=True):
        """The named official config + its checkpoint, EMA weights in, on ``device``.

        ``allow_tf32=False`` runs convolutions in full float32 (PyTorch's default on Ampere+ is
        TF32 for cuDNN convolutions, which the original 2021 runs did not have)."""
        config = get_config(config_name)
        if torch.device(device).type == "cuda":
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        sde, eps = build_sde(config)
        model, info = load_score_model(config, checkpoint, device=device, use_ema=use_ema)
        score_fn = mutils.get_score_fn(sde, model, train=False,
                                       continuous=config.training.continuous)
        shape = (config.data.num_channels, config.data.image_size, config.data.image_size)
        info.update(config_name=config_name, allow_tf32=bool(allow_tf32),
                    upstream_commit=UPSTREAM_COMMIT)
        return cls(sde, score_fn, shape, device=device, eps=eps, config=config, model=model,
                   info=info, deterministic_cudnn=deterministic_cudnn)

    def describe(self):
        """Everything a results file needs to say which teacher produced it."""
        sde = self.sde
        d = dict(kind=type(sde).__name__, N=int(sde.N), T=float(sde.T), eps=self.eps,
                 device=str(self.device), image_shape=self.image_shape, **self.info)
        for k in ("sigma_min", "sigma_max", "beta_0", "beta_1"):
            if hasattr(sde, k):
                d[k] = float(getattr(sde, k))
        if self.config is not None:
            d["config"] = to_dict(self.config)
        return d

    # ------------------------------------------------------------------ the clock
    def grid_index(self, t):
        """Nearest index of ``t`` on the official grid and the grid time it snaps to."""
        t = float(t)
        tol = 1e-6
        if not (self.eps - tol <= t <= self.sde.T + tol):
            raise ValueError(f"t={t} outside the sampler's time range [{self.eps}, {self.sde.T}]")
        i = int(np.abs(self._grid - t).argmin())
        return i, float(self._grid[i])

    def transition_indices(self, t, s):
        """``(i_t, i_s, t_grid, s_grid)`` for a transition ``t -> s``; requires ``0 < s < t``."""
        if not float(s) < float(t):
            raise ValueError(f"need s < t, got t={t}, s={s}")
        i_t, tg = self.grid_index(t)
        i_s, sg = self.grid_index(s)
        if i_s <= i_t:
            raise ValueError(f"t={t} and s={s} snap to the same grid point (index {i_t}); "
                             f"grid spacing is {(self.sde.T - self.eps) / (self.sde.N - 1):.2e}")
        return i_t, i_s, tg, sg

    def sigma(self, t):
        """Marginal std of ``p_t(x | x_0)`` (``sigma(t)`` for VE)."""
        vec = torch.as_tensor([float(t)], device=self.device)
        z = torch.zeros((1, *self.image_shape), device=self.device)
        return float(self.sde.marginal_prob(z, vec)[1].reshape(-1)[0])

    def mean_coeff(self, t):
        """Coefficient of ``x_0`` in the marginal mean (1 for VE, ``exp(...)`` for VP)."""
        vec = torch.as_tensor([float(t)], device=self.device)
        o = torch.ones((1, *self.image_shape), device=self.device)
        return float(self.sde.marginal_prob(o, vec)[0].reshape(-1)[0])

    # ------------------------------------------------------------------ pieces
    def _as_batch(self, x, name):
        x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x)
        x = x.to(self.device, torch.float32)
        if x.ndim == len(self.image_shape):
            x = x[None]
        if x.ndim != len(self.image_shape) + 1 or tuple(x.shape[1:]) != self.image_shape:
            raise ValueError(f"{name} must have shape (B,)+{self.image_shape}, got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains non-finite values")
        return x

    def score(self, x, t):
        """``grad_x log p_t(x)`` for a batch ``x`` at one time ``t``; counts the evaluation."""
        x = self._as_batch(x, "x")
        vec = torch.full((x.shape[0],), float(t), device=self.device)
        self.n_score_calls += 1
        with torch.no_grad():
            return self._score_fn(x, vec)

    def forward_noise(self, x0, t, seed):
        """``x_t ~ p_t(. | x_0)`` from the forward SDE's marginal: ``mean(x_0, t) + std(t) z``.

        ``z`` comes from a CPU generator seeded with ``seed`` so the frozen ``x_t`` is identical on
        any device.  Returns ``(x_t, info)`` with ``x_t`` shaped like ``x0`` (no batch axis)."""
        x0 = self._as_batch(x0, "x0")
        i, tg = self.grid_index(t)
        vec = torch.full((x0.shape[0],), tg, device=self.device)
        mean, std = self.sde.marginal_prob(x0, vec)
        g = torch.Generator().manual_seed(int(seed))
        z = torch.randn(x0.shape, generator=g).to(self.device)
        std = std.reshape(-1, *([1] * len(self.image_shape))) if torch.is_tensor(std) else std
        x_t = mean + std * z
        info = dict(t=tg, i_t=i, sigma=float(torch.as_tensor(std).reshape(-1)[0]),
                    forward_noise_seed=int(seed))
        return x_t[0], info

    def tweedie_denoise(self, x, t):
        """``E_theta[x_0 | x_t = x] = (x + std(t)^2 score(x, t)) / mean_coeff(t)`` -- Tweedie.

        A deterministic function of ``x``, one network evaluation; used for visualisation and as
        the input of the feature extractor (a noisy ``x_s`` at sigma ~ 1 is unreadable).  NOT part
        of the official sampler except at its very last step."""
        x = self._as_batch(x, "x")
        i, tg = self.grid_index(t)
        s = self.score(x, tg)
        return (x + self.sigma(tg) ** 2 * s) / self.mean_coeff(tg)

    @contextlib.contextmanager
    def _noise_from(self, generators):
        """Route the sampler's Gaussian noise through per-sample generators."""
        def hook(x):
            out = torch.empty_like(x)
            for i, g in enumerate(generators):
                out[i] = torch.randn(x.shape[1:], generator=g, device=x.device, dtype=x.dtype)
            return out
        old = sampling.randn_like
        sampling.randn_like = hook
        try:
            yield
        finally:
            sampling.randn_like = old

    def _generators(self, seeds):
        return [torch.Generator(device=self.device).manual_seed(int(s)) for s in seeds]

    # ------------------------------------------------------------------ the transition
    def sample_transition(self, x_t, t, s, seeds, *, predictor="reverse_diffusion",
                          corrector="none", snr=0.075, n_steps_each=1, probability_flow=False,
                          record_every=0, record_at=()):
        """Independent rollouts ``x_s^(i) ~ p_theta(x_s | x_t)``, one per seed, from ONE ``x_t``.

        The loop is ``sampling.get_pc_sampler``'s loop restricted to grid indices ``[i_t, i_s)``:
        at each level ``t_i`` the corrector updates ``x`` (``n_steps_each`` Langevin steps, or
        nothing), then the predictor moves it to ``t_{i+1}``.  After the loop ``x`` sits at
        ``timesteps[i_s]``.  Rollout ``i`` draws every noise tensor from its own generator seeded
        with ``seeds[i]``; the Langevin step size, as upstream, uses the batch-mean gradient and
        noise norms, so a rollout is reproducible given its seed AND the batch it ran in.

        ``record_every`` / ``record_at`` snapshot the state every so many steps / after the
        predictor step that lands on the given grid indices (``intermediate_steps`` names them).
        Because each rollout consumes its own noise stream in step order, the snapshot of the
        state at grid index ``j`` is exactly the rollout one would get from ``x_t`` to
        ``timesteps[j]`` with the same seed -- the short-transition control comes for free.
        """
        if predictor not in PREDICTORS:
            raise ValueError(f"predictor must be one of {PREDICTORS}, got {predictor!r}")
        if corrector not in CORRECTORS:
            raise ValueError(f"corrector must be one of {CORRECTORS}, got {corrector!r}")
        if probability_flow and corrector != "none":
            raise ValueError("the probability-flow ODE control takes corrector='none'")
        if isinstance(seeds, (int, np.integer)):
            seeds = (int(seeds),)
        seeds = tuple(int(v) for v in seeds)
        if not seeds:
            raise ValueError("need at least one rollout seed")
        i_t, i_s, tg, sg = self.transition_indices(t, s)
        record_at = {int(j) for j in record_at}
        if any(j <= i_t or j > i_s for j in record_at):
            raise ValueError(f"record_at indices must lie in ({i_t}, {i_s}], got {sorted(record_at)}")

        x_t = self._as_batch(x_t, "x_t")
        if x_t.shape[0] != 1:
            raise ValueError("sample_transition takes ONE x_t (shape (C,H,W) or (1,C,H,W)); "
                             "batch several rollouts through `seeds`")
        B = len(seeds)
        x = x_t.detach().clone().expand(B, *self.image_shape).contiguous()   # x_t itself untouched
        x_mean = x

        calls0 = self.n_score_calls

        def score_fn(xx, tt):
            self.n_score_calls += 1
            return self._score_fn(xx, tt)

        P = sampling.get_predictor(predictor)(self.sde, score_fn, probability_flow)
        C = sampling.get_corrector(corrector)(self.sde, score_fn, snr, n_steps_each)
        snaps, snap_i = [], []
        t0 = time.perf_counter()
        with torch.no_grad(), self._noise_from(self._generators(seeds)):
            for i in range(i_t, i_s):
                vec_t = torch.ones(B, device=self.device) * self.timesteps[i]
                x, x_mean = C.update_fn(x, vec_t)
                x, x_mean = P.update_fn(x, vec_t)
                if (record_every and (((i + 1 - i_t) % record_every == 0) or i == i_s - 1)) \
                        or (i + 1) in record_at:
                    snaps.append(x.detach().cpu().numpy().astype(np.float32))
                    snap_i.append(i + 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        runtime = time.perf_counter() - t0
        n_calls = self.n_score_calls - calls0
        return Transition(
            x_s=x, x_s_mean=x_mean, t=tg, s=sg, i_t=i_t, i_s=i_s, n_steps=i_s - i_t,
            nfe_per_sample=n_calls, n_forward_calls=n_calls, seeds=seeds,
            method=dict(predictor=predictor, corrector=corrector, snr=float(snr),
                        n_steps_each=int(n_steps_each), probability_flow=bool(probability_flow),
                        grid_N=int(self.sde.N), eps=self.eps),
            runtime_s=runtime,
            intermediates=np.stack(snaps) if snaps else None,
            intermediate_steps=np.asarray(snap_i, dtype=int) if snaps else None,
            intermediate_times=self._grid[snap_i] if snaps else None,
        )

    def ode_transition(self, x_t, t, s, *, predictor="reverse_diffusion", n_copies=1):
        """Deterministic control: the probability-flow ODE from ``x_t`` to ``s`` on the same grid.

        No noise is drawn, so the seeds are irrelevant; ``n_copies`` identical copies are run in
        one batch to expose any nondeterminism of the kernels themselves."""
        return self.sample_transition(x_t, t, s, seeds=tuple(range(n_copies)),
                                      predictor=predictor, corrector="none",
                                      probability_flow=True)
