"""The complete teacher trajectory as ten contiguous intervals, and teacher-side operations on it.

The teacher is the official predictor-only reverse-diffusion sampler with noise removal
(:mod:`gmmd.pipeline`):

    x = sigma_max z
    for i in range(N):  x, x_mean = predictor(x, timesteps[i])
    return x_mean

After steps ``[0, i)`` the state sits at grid index ``i`` (time ``timesteps[i]``, noise level
``discrete_sigmas[N-1-i]``).  With boundaries ``b_j = round(j N / n)``, interval ``j`` is the
teacher steps ``[b_j, b_{j+1})``:

* ``j < n-1``: from the stochastic state at grid index ``b_j`` to the stochastic state at
  ``b_{j+1}``;
* ``j = n-1`` (terminal): steps ``[b_{n-1}, N-1)`` reach the last stochastic state, at grid
  index ``N-1`` (time ``eps``, ``sigma_min``).  Step ``N-1`` is then deterministic under noise
  removal: ``x_0 = x + sigma_min^2 s_theta(x, eps)``, the teacher's own Tweedie denoiser
  (:func:`terminal_denoise`), and ``x_0`` is the final image representation.

Nothing here changes the teacher's discretisation, time direction, prior or output: every
teacher-side state is produced by :func:`gmmd.pipeline.fine_steps`, i.e. by the vendored
predictor itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from gmmd import pipeline as P

__all__ = ["Interval", "make_partition", "partition_table", "seed_of", "SeededGenerators",
           "prior_from_seeds", "trajectory_states", "interval_rollouts", "terminal_denoise",
           "coarse_trajectory", "COARSE_KINDS"]

COARSE_KINDS = ("reverse_diffusion", "ancestral")


@dataclass(frozen=True)
class Interval:
    j: int
    step_start: int            # first teacher step = grid index of the start state
    step_end: int              # one past the last teacher step
    n_steps: int
    terminal: bool
    grid_end_state: int        # grid index of the END STATE the IBW target is defined at
    t_start: float             # grid time of the start state
    t_end_state: float         # grid time of that end state
    sigma_start: float         # sigma(t) as the score network sees it (continuous VE sigma)
    sigma_end_state: float
    gap: float                 # sigma_start^2 - sigma_end_state^2: the forward-kernel variance
    v_anc: float               # sigma_end^2 * gap / sigma_start^2 (Gaussian-data posterior variance)
    mean_gap: float            # E[end | start] = x + mean_gap * score(x, t_start): gap, or
    #                            sigma_start^2 for the terminal interval, whose output is at sigma 0
    discrete_sigma_start: float
    discrete_sigma_end: float  # noise level after the interval's last step (0: terminal output)


def make_partition(teacher, n_intervals=10):
    """The complete grid ``[0, N)`` split into ``n_intervals`` contiguous step ranges."""
    N = int(teacher.sde.N)
    b = np.linspace(0, N, int(n_intervals) + 1).round().astype(int)
    if len(set(b.tolist())) != len(b):
        raise ValueError(f"cannot split {N} steps into {n_intervals} non-empty intervals")
    ds = teacher.sde.discrete_sigmas.detach().double().cpu().numpy()
    grid = teacher._grid
    out = []
    for j in range(int(n_intervals)):
        s0, s1 = int(b[j]), int(b[j + 1])
        terminal = s1 == N
        g_end = N - 1 if terminal else s1
        t0, t1 = float(grid[s0]), float(grid[g_end])
        sig0, sig1 = teacher.sigma(t0), teacher.sigma(t1)
        gap = sig0 ** 2 - sig1 ** 2
        out.append(Interval(
            j=j, step_start=s0, step_end=s1, n_steps=s1 - s0, terminal=terminal,
            grid_end_state=g_end, t_start=t0, t_end_state=t1, sigma_start=sig0,
            sigma_end_state=sig1, gap=gap, v_anc=sig1 ** 2 * gap / sig0 ** 2,
            mean_gap=sig0 ** 2 if terminal else gap,
            discrete_sigma_start=float(ds[N - 1 - s0]),
            discrete_sigma_end=0.0 if terminal else float(ds[N - 1 - s1])))
    return out


def partition_table(parts):
    return [asdict(iv) for iv in parts]


# ======================================================================================
# seeds
# ======================================================================================
def seed_of(*keys):
    """A reproducible 63-bit seed from a tuple of non-negative integers (independent streams)."""
    return int(np.random.SeedSequence([int(k) for k in keys]).generate_state(1, dtype=np.uint64)[0]
               >> np.uint64(1))


class SeededGenerators:
    """One device generator per row, seeded explicitly; the interface of ``pipeline.Generators``."""

    def __init__(self, seeds, device):
        self.seeds = [int(s) for s in seeds]
        self.device = torch.device(device)
        self.gens = [torch.Generator(device=self.device).manual_seed(s) for s in self.seeds]

    def randn(self, shape, dtype=torch.float32):
        return torch.stack([torch.randn(tuple(shape), generator=g, device=self.device, dtype=dtype)
                            for g in self.gens])


def prior_from_seeds(image_shape, sigma_max, seeds, device):
    """``x_T = sigma_max z``, ``z`` from a CPU generator seeded with the image seed.

    Identical to :func:`gmmd.pipeline.prior_sample` (which exp4's teacher images used), but
    needs no teacher object -- the standalone sampler calls this."""
    zs = [torch.randn(tuple(image_shape), generator=torch.Generator().manual_seed(int(s)))
          for s in seeds]
    return (torch.stack(zs) * float(sigma_max)).to(device)


# ======================================================================================
# teacher-side rollouts
# ======================================================================================
def trajectory_states(teacher, seeds, parts, batch_size=64, traj_seeds=None):
    """Teacher trajectories from ``x_T`` (one per image seed), recording every interval's start.

    Returns ``(states, nfe)``, ``states`` (n_intervals, B, C, H, W) float32 on the CPU: the state
    at grid indices ``b_0 = 0, b_1, ..., b_{n-1}``.  The trajectory noise of image ``k`` comes from
    one generator seeded ``traj_seeds[k]`` and consumed in step order, so splitting the loop at
    the boundaries is the uninterrupted sampler (``tests/test_exp5.py``)."""
    if traj_seeds is None:
        traj_seeds = [seed_of(51, s) for s in seeds]
    gens = SeededGenerators(traj_seeds, teacher.device)
    x = P.prior_sample(teacher, seeds)
    states, nfe = [x.detach().cpu()], 0
    for iv in parts[:-1]:
        x, _, n = P.fine_steps(teacher, x, iv.step_start, iv.step_end, gens, batch_size)
        states.append(x.detach().cpu())
        nfe += n
    return torch.stack(states), nfe


def interval_rollouts(teacher, x_start, iv, rollout_seeds, batch_size=64):
    """Teacher rollouts of interval ``iv``: ``x_start`` (A, C, H, W), ``rollout_seeds`` (A, R).

    Row ``(a, r)`` starts from ``x_start[a]`` and draws its noise from its own generator, so it
    does not depend on the batch.  Returns ``(end (A, R, C, H, W) on device, nfe)``; for the
    terminal interval ``end`` is the noise-removed output ``x_0`` (the last predictor mean)."""
    rollout_seeds = np.asarray(rollout_seeds, dtype=np.int64)
    A, R = rollout_seeds.shape
    x = x_start.to(teacher.device)[:, None].expand(A, R, *teacher.image_shape)
    x = x.reshape(A * R, *teacher.image_shape).contiguous()
    gens = SeededGenerators(rollout_seeds.reshape(-1).tolist(), teacher.device)
    x, x_mean, nfe = P.fine_steps(teacher, x, iv.step_start, iv.step_end, gens, batch_size)
    end = x_mean if iv.terminal else x
    return end.reshape(A, R, *teacher.image_shape), nfe


def terminal_denoise(teacher, x, batch_size=64):
    """The teacher's last step under noise removal, ``x + G^2 s_theta(x, eps)`` with
    ``G = discrete_sigmas[0] = sigma_min``: the ``x_mean`` of ``ReverseDiffusionPredictor`` at
    ``timesteps[N-1]``, evaluated through the vendored reverse-SDE discretisation.

    Returns ``(x_0, nfe)``.  Deterministic -- no noise is drawn."""
    N = int(teacher.sde.N)
    nfe = 0

    def score_fn(xx, tt):
        nonlocal nfe
        out = torch.empty_like(xx)
        for j in range(0, xx.shape[0], batch_size):
            out[j:j + batch_size] = teacher._score_fn(xx[j:j + batch_size], tt[j:j + batch_size])
        nfe += int(xx.shape[0])
        return out

    rsde = teacher.sde.reverse(score_fn, False)
    x = x.to(teacher.device, torch.float32)
    vec = torch.ones(x.shape[0], device=x.device) * teacher.timesteps[N - 1]
    with torch.no_grad():
        f, _ = rsde.discretize(x, vec)
    return x - f, nfe


# ======================================================================================
# coarse teacher baselines over the complete trajectory
# ======================================================================================
def coarse_trajectory(teacher, seeds, n_steps, kind="reverse_diffusion", batch_size=64,
                      noise_seeds=None):
    """``n_steps`` teacher steps over the complete grid ``[0, N)``; the final mean is returned.

    Blocks ``[a, b)`` split the grid as evenly as possible (:func:`gmmd.pipeline.coarse_blocks`).
    Each block is ONE application of a vendored predictor's rule with a longer stride, the
    score taken at ``timesteps[a]``:

    * ``reverse_diffusion`` (SMLD): ``x_mean = x + G^2 s``, ``x = x_mean + G z``,
      ``G^2 = sigma_a^2 - sigma_b^2``;
    * ``ancestral`` (``AncestralSamplingPredictor.vesde_update_fn``): same mean,
      ``std = sqrt(sigma_b^2 (sigma_a^2 - sigma_b^2) / sigma_a^2)``,

    with ``sigma_a`` the noise level at the block start and ``sigma_b`` after the block (0 for
    the last block, which therefore ends in the teacher's noise-removal convention: its mean
    is the Tweedie denoiser at ``sigma_a``).  Returns ``(x_0, nfe)``."""
    if kind not in COARSE_KINDS:
        raise ValueError(f"kind must be one of {COARSE_KINDS}, got {kind!r}")
    N = int(teacher.sde.N)
    if noise_seeds is None:
        noise_seeds = [seed_of(52, COARSE_KINDS.index(kind), int(n_steps), s) for s in seeds]
    gens = SeededGenerators(noise_seeds, teacher.device)
    x = P.prior_sample(teacher, seeds)
    x_mean, nfe = x, 0
    with torch.no_grad():
        for a, b in P.coarse_blocks(0, N, n_steps):
            sig_a, _ = P.step_sigmas(teacher, a)
            _, sig_b = P.step_sigmas(teacher, b - 1)
            s, n = P._score(teacher, x, teacher.timesteps[a], batch_size)
            nfe += n
            g2 = sig_a ** 2 - sig_b ** 2
            x_mean = x + g2 * s
            std = np.sqrt(g2) if kind == "reverse_diffusion" else np.sqrt(sig_b ** 2 * g2 / sig_a ** 2)
            x = x_mean + float(std) * gens.randn(x.shape[1:]).to(x.dtype)
    return x_mean, nfe
