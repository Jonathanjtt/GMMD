"""The official ``score_sde_pytorch`` configs the teacher can run, as plain attribute trees.

Upstream builds these with ``ml_collections.ConfigDict``; the values below are copied VERBATIM
from the named upstream files (commit ``_vendor.provenance.UPSTREAM_COMMIT``) into nested
:class:`types.SimpleNamespace` objects, which is all the vendored model code reads
(``config.model.*``, ``config.data.*``, ``config.training.continuous``).  ``ml_collections`` is
therefore not a dependency.  Only the fields the model, SDE and sampler consume are kept; the
training/optimiser/evaluation blocks are irrelevant to a frozen checkpoint and omitted.

Time convention (the one thing this repo has to import from Song et al., since the benchmark
never had a diffusion clock): continuous ``t in [eps, T=1]``; ``t = 1`` is the prior
``N(0, sigma_max^2 I)`` for the VE SDE, ``t -> 0`` the data.  The forward VE marginal is
``x_t = x_0 + sigma(t) z`` with ``sigma(t) = sigma_min (sigma_max / sigma_min)^t``.  Sampling
walks the grid ``torch.linspace(T, eps, N)`` downwards with ``eps = 1e-5`` for VE models.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

__all__ = ["CONFIGS", "get_config", "available", "to_dict"]


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def to_dict(ns):
    """Inverse of the namespace build, for provenance dumps."""
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    return copy.deepcopy(ns)


# configs/default_lsun_configs.py  (get_default_configs), relevant blocks only
_LSUN_DEFAULTS = dict(
    training=dict(continuous=True, likelihood_weighting=False, reduce_mean=False),
    sampling=dict(n_steps_each=1, noise_removal=True, probability_flow=False, snr=0.075),
    data=dict(dataset="LSUN", image_size=256, random_flip=True, uniform_dequantization=False,
              centered=False, num_channels=3),
    model=dict(sigma_max=378, sigma_min=0.01, num_scales=2000, beta_min=0.1, beta_max=20.,
               dropout=0., embedding_type="fourier"),
)


def _celebahq_256_ncsnpp_continuous():
    # configs/ve/celebahq_256_ncsnpp_continuous.py  (get_config)
    c = copy.deepcopy(_LSUN_DEFAULTS)
    c["training"].update(sde="vesde", continuous=True)
    c["sampling"].update(method="pc", predictor="reverse_diffusion", corrector="langevin")
    c["data"].update(dataset="CelebAHQ", image_size=256)
    c["model"].update(
        name="ncsnpp", sigma_max=348, scale_by_sigma=True, ema_rate=0.999,
        normalization="GroupNorm", nonlinearity="swish", nf=128, ch_mult=(1, 1, 2, 2, 2, 2, 2),
        num_res_blocks=2, attn_resolutions=(16,), resamp_with_conv=True, conditional=True,
        fir=True, fir_kernel=[1, 3, 3, 1], skip_rescale=True, resblock_type="biggan",
        progressive="output_skip", progressive_input="input_skip", progressive_combine="sum",
        attention_type="ddpm", init_scale=0., fourier_scale=16, conv_size=3)
    return c


# The official CelebA-family checkpoint of score_sde_pytorch is THIS one: the README states the
# original CelebA (64x64) and CelebA-HQ checkpoints could not be released, and offers a
# re-trained CelebA-HQ 256px model instead (Google Drive folder 19VJ7UZTE-ytGX6z5rl-tumW9c0Ps3itk,
# file checkpoint_48.pth).  No 64x64 CelebA checkpoint exists upstream.
CONFIGS = {
    "ve/celebahq_256_ncsnpp_continuous": _celebahq_256_ncsnpp_continuous,
}

# Sampling eps per SDE family, from run_lib.py (the reverse SDE is integrated to eps, not 0).
SAMPLING_EPS = {"vesde": 1e-5, "vpsde": 1e-3, "subvpsde": 1e-3}


def available():
    return sorted(CONFIGS)


def get_config(name):
    """The named official config as a namespace tree (a fresh copy each call)."""
    if name not in CONFIGS:
        raise KeyError(f"unknown score_sde config {name!r}; available: {available()}")
    return _ns(CONFIGS[name]())
