"""Full-image generation split into prefix / segment / suffix, for single-interval replacement.

The official sampler (``sampling.get_pc_sampler`` with predictor ``reverse_diffusion`` and
corrector ``none``) is the loop

    x = sigma_max * z
    for i in range(N):  x, x_mean = predictor(x, timesteps[i])
    return x_mean                                     # noise_removal = True

Step ``i`` uses the score at ``timesteps[i]`` and moves the noise level from
``discrete_sigmas[k_i]`` to ``discrete_sigmas[k_i - 1]`` with ``k_i = floor(t_i (N-1))``.
After the steps ``[0, i)`` have run, the state "sits at" grid index ``i``.  Splitting the loop
at ``i_t < i_s`` into ``[0, i_t)``, ``[i_t, i_s)``, ``[i_s, N)`` therefore changes nothing: every
step runs exactly once, each in exactly one segment, and with no corrector there is no boundary
update to assign.  ``tests/test_interval_replacement.py`` checks the split is bit-identical to the
uninterrupted loop.

Randomness.  Every image owns one CPU-seeded prior draw and one CUDA generator per stage
(prefix, segment, suffix), seeded from the image's seed and the stage name, so an image's
trajectory does not depend on which batch it ran in, and two methods can share the prefix and
suffix streams while drawing their segment independently.

The coarse teacher takes ``n`` larger SMLD steps over the same interval: a block of fine steps
``[a, b)`` becomes one step from ``discrete_sigmas[k_a]`` to ``discrete_sigmas[k_{b-1} - 1]``
with the score at ``timesteps[a]`` -- the same update rule, a longer stride.  With blocks of one
step it is the fine sampler again (checked in the tests).
"""

from __future__ import annotations

import time

import numpy as np
import torch

__all__ = ["STAGES", "stage_seed", "Generators", "prior_sample", "step_sigmas", "fine_steps",
           "coarse_steps", "coarse_blocks", "Timer"]

STAGES = ("prefix", "segment", "suffix", "gmmd_fit", "gmmd_sample", "coarse")


def stage_seed(image_seed, stage):
    """A distinct, reproducible 63-bit seed per (image, stage)."""
    return int(np.random.SeedSequence([int(image_seed), STAGES.index(stage)])
               .generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


class Generators:
    """One CUDA generator per image for one stage; ``randn(shape)`` draws one tensor per image."""

    def __init__(self, image_seeds, stage, device):
        self.gens = [torch.Generator(device=device).manual_seed(stage_seed(s, stage))
                     for s in image_seeds]
        self.device = torch.device(device)

    def randn(self, shape, dtype=torch.float32):
        return torch.stack([torch.randn(tuple(shape), generator=g, device=self.device, dtype=dtype)
                            for g in self.gens])


def prior_sample(teacher, image_seeds):
    """``x_T = sigma_max z`` with ``z`` from a CPU generator per image (device-independent)."""
    zs = [torch.randn(teacher.image_shape, generator=torch.Generator().manual_seed(int(s)))
          for s in image_seeds]
    return (torch.stack(zs) * float(teacher.sde.sigma_max)).to(teacher.device)


def step_sigmas(teacher, i):
    """``(sigma, adjacent_sigma)`` of fine step ``i``, computed as ``VESDE.discretize`` does."""
    sde = teacher.sde
    t = teacher.timesteps[i].reshape(1)
    k = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[k]
    adj = torch.where(k == 0, torch.zeros_like(t), sde.discrete_sigmas[(k - 1).clamp(min=0)])
    return float(sigma), float(adj)


def _score(teacher, x, t, batch_size):
    """Score at one grid time for a batch, chunked; returns (score, examples evaluated)."""
    out = torch.empty_like(x)
    vec_full = torch.full((x.shape[0],), float(t), device=x.device)
    for j in range(0, x.shape[0], batch_size):
        out[j:j + batch_size] = teacher._score_fn(x[j:j + batch_size], vec_full[j:j + batch_size])
    return out, int(x.shape[0])


def fine_steps(teacher, x, i0, i1, gens, batch_size=64):
    """The official reverse-diffusion predictor over grid steps ``[i0, i1)``, one noise stream per row.

    Uses the vendored predictor itself (``ReverseDiffusionPredictor.update_fn``) with the
    teacher's per-sample noise hook, so the arithmetic is upstream's, not a re-implementation.
    Returns ``(x, x_mean, nfe)`` with ``nfe`` = score-network examples evaluated.
    """
    from gmmd.teachers.score_sde._vendor import sampling

    nfe = 0

    def score_fn(xx, tt):
        nonlocal nfe
        out = torch.empty_like(xx)
        for j in range(0, xx.shape[0], batch_size):
            out[j:j + batch_size] = teacher._score_fn(xx[j:j + batch_size], tt[j:j + batch_size])
        nfe += int(xx.shape[0])
        return out

    P = sampling.get_predictor("reverse_diffusion")(teacher.sde, score_fn, False)
    x_mean = x
    B = x.shape[0]
    with torch.no_grad(), teacher._noise_from(gens.gens):
        for i in range(i0, i1):
            vec_t = torch.ones(B, device=x.device) * teacher.timesteps[i]
            x, x_mean = P.update_fn(x, vec_t)
    return x, x_mean, nfe


def coarse_blocks(i0, i1, n_steps):
    """Split ``[i0, i1)`` into ``n_steps`` contiguous blocks as evenly as possible."""
    edges = np.linspace(i0, i1, int(n_steps) + 1).round().astype(int)
    if len(set(edges.tolist())) != len(edges):
        raise ValueError(f"cannot split [{i0}, {i1}) into {n_steps} non-empty blocks")
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


def coarse_steps(teacher, x, i0, i1, n_steps, gens, batch_size=64):
    """``n_steps`` SMLD steps covering fine steps ``[i0, i1)``; same rule, longer strides.

    Block ``[a, b)``: ``x_mean = x + G^2 s(x, t_a)``, ``x = x_mean + G z`` with
    ``G^2 = sigma(step a)^2 - adjacent_sigma(step b-1)^2``.  Returns ``(x, x_mean, nfe)``.
    """
    nfe = 0
    x_mean = x
    with torch.no_grad():
        for a, b in coarse_blocks(i0, i1, n_steps):
            sig_a, _ = step_sigmas(teacher, a)
            _, adj_b = step_sigmas(teacher, b - 1)
            G = torch.sqrt(torch.tensor(sig_a ** 2 - adj_b ** 2, device=x.device,
                                        dtype=torch.float32))
            s, n = _score(teacher, x, teacher.timesteps[a], batch_size)
            nfe += n
            x_mean = x + (G ** 2) * s
            x = x_mean + G * gens.randn(x.shape[1:]).to(x.dtype)
    return x, x_mean, nfe


class Timer:
    """Wall-clock and peak CUDA memory of one stage, with the device synchronised at both ends."""

    def __init__(self, device):
        self.device = torch.device(device)

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_gb = torch.cuda.max_memory_allocated(self.device) / 2 ** 30
        else:
            self.peak_gb = float("nan")
        self.seconds = time.perf_counter() - self.t0
        return False
