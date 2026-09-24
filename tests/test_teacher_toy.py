"""The GMMD teacher on an ANALYTIC target: no network, no checkpoint, CPU only.

The data is Gaussian, ``x_0 ~ N(mu, Sigma)`` in d = 2 laid out as (2, 1, 1) "images", so the
VE marginal score is exact, ``score(x, t) = -(Sigma + sigma(t)^2 I)^{-1} (x - mu)``, and the
reverse-diffusion (SMLD) predictor is an affine Gaussian recursion whose transition kernel from
``x_t`` to ``x_s`` can be propagated in closed form.  That gives an exact check of the sampler,
of the fixed-``x_t`` semantics, of the seeding, and of the deterministic control -- and a
numerical look at how close the discretised kernel is to the exact Bayes posterior
``p(x_s | x_t) ∝ p_s(x_s) N(x_t; x_s, (sigma_t^2 - sigma_s^2) I)`` (the continuous-time limit).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gmmd.teachers.score_sde._vendor import sde_lib
from gmmd.teachers.score_sde.teacher import ScoreSDETeacher

MU = np.array([1.0, -1.0])
SIGMA = np.diag([0.5, 2.0])
SMIN, SMAX, N_GRID = 0.01, 348.0, 400


def _sigma(t):
    return SMIN * (SMAX / SMIN) ** t


def make_teacher(N=N_GRID, device="cpu"):
    sde = sde_lib.VESDE(sigma_min=SMIN, sigma_max=SMAX, N=N)
    mu = torch.tensor(MU, dtype=torch.float32).view(1, 2, 1, 1)
    Sig = torch.tensor(SIGMA, dtype=torch.float32)

    def score_fn(x, t):                                   # (B,2,1,1), (B,) -> (B,2,1,1)
        s2 = (SMIN * (SMAX / SMIN) ** t) ** 2             # (B,)
        out = torch.empty_like(x)
        for b in range(x.shape[0]):
            P = torch.linalg.inv(Sig + s2[b] * torch.eye(2))
            out[b] = -(P @ (x[b] - mu[0]).reshape(2, 1)).reshape(2, 1, 1)
        return out

    return ScoreSDETeacher(sde, score_fn, image_shape=(2, 1, 1), device=device, eps=1e-5)


def exact_kernel(teacher, x_t, i_t, i_s):
    """Mean / cov of x_s given x_t under the reverse-diffusion predictor (affine recursion)."""
    sde = teacher.sde
    grid = teacher.timesteps.cpu().numpy().astype(np.float64)
    ds = sde.discrete_sigmas.cpu().numpy().astype(np.float64)
    m = np.asarray(x_t, float).reshape(2).copy()
    Cv = np.zeros((2, 2))
    for i in range(i_t, i_s):
        t = float(torch.tensor(grid[i], dtype=torch.float32))      # the sampler sees float32 times
        k = int(torch.tensor(t * (sde.N - 1) / sde.T).long())      # sde_lib.VESDE.discretize
        G2 = ds[k] ** 2 - (ds[k - 1] ** 2 if k > 0 else 0.0)
        P = np.linalg.inv(SIGMA + _sigma(t) ** 2 * np.eye(2))       # score at the continuous time
        A = np.eye(2) - G2 * P
        m = A @ m + G2 * P @ MU
        Cv = A @ Cv @ A.T + G2 * np.eye(2)
    return m, Cv


def bayes_posterior(x_t, st, ss):
    """Exact p(x_s | x_t) for Gaussian data under the time reversal (continuous limit)."""
    Ss = SIGMA + ss ** 2 * np.eye(2)
    Lam = np.linalg.inv(Ss) + np.eye(2) / (st ** 2 - ss ** 2)
    Cv = np.linalg.inv(Lam)
    m = Cv @ (np.linalg.solve(Ss, MU) + np.asarray(x_t, float).reshape(2) / (st ** 2 - ss ** 2))
    return m, Cv


@pytest.fixture(scope="module")
def teacher():
    return make_teacher()


@pytest.fixture(scope="module")
def x_t(teacher):
    x0 = torch.tensor([1.3, -0.4], dtype=torch.float32).view(2, 1, 1)
    xt, info = teacher.forward_noise(x0, 0.6, seed=0)
    return xt


def test_forward_noise_is_reproducible_and_seed_sensitive(teacher):
    x0 = torch.tensor([1.3, -0.4], dtype=torch.float32).view(2, 1, 1)
    a, ia = teacher.forward_noise(x0, 0.6, seed=0)
    b, _ = teacher.forward_noise(x0, 0.6, seed=0)
    c, _ = teacher.forward_noise(x0, 0.6, seed=1)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert a.shape == (2, 1, 1) and a.dtype == torch.float32
    assert ia["sigma"] == pytest.approx(_sigma(ia["t"]), rel=1e-5)
    # x_t = x_0 + sigma z with the SAME z for the same seed at another t
    d, idd = teacher.forward_noise(x0, 0.7, seed=0)
    z1, z2 = (a - x0) / ia["sigma"], (d - x0) / idd["sigma"]
    assert torch.allclose(z1, z2, atol=1e-5)


def test_time_grid_matches_upstream_index_mapping(teacher):
    # the reverse-diffusion predictor maps grid point i to discrete-sigma index N-1-i
    sde = teacher.sde
    idx = (teacher.timesteps * (sde.N - 1) / sde.T).long().cpu().numpy()
    assert np.array_equal(idx, np.arange(sde.N - 1, -1, -1))


def test_timestep_validity(teacher, x_t):
    with pytest.raises(ValueError):
        teacher.sample_transition(x_t, 0.5, 0.6, seeds=0)         # s > t
    with pytest.raises(ValueError):
        teacher.sample_transition(x_t, 0.5, 0.5, seeds=0)         # s == t
    with pytest.raises(ValueError):
        teacher.sample_transition(x_t, 1.2, 0.5, seeds=0)         # t > T
    with pytest.raises(ValueError):
        teacher.sample_transition(x_t, 0.5, -0.1, seeds=0)        # s < 0
    with pytest.raises(ValueError):
        teacher.sample_transition(x_t, 0.5, 0.5 - 1e-9, seeds=0)  # same grid point
    with pytest.raises(ValueError):
        teacher.sample_transition(x_t, 0.6, 0.5, seeds=0, probability_flow=True, corrector="langevin")
    with pytest.raises(ValueError):
        teacher.sample_transition(x_t, 0.6, 0.5, seeds=0, predictor="nope")


def test_x_t_is_frozen_and_batched_rollouts_start_from_it(teacher, x_t):
    before = x_t.clone()
    res = teacher.sample_transition(x_t, 0.6, 0.55, seeds=(1, 2, 3), record_every=1)
    assert torch.equal(x_t, before)
    # the first snapshot is one predictor step from x_t for EVERY rollout
    assert res.intermediates.shape[0] == res.n_steps
    assert res.intermediates.shape[1:] == (3, 2, 1, 1)
    # a single-noise-step distance from x_t is small relative to |x_t| but nonzero and differs by seed
    first = res.intermediates[0]
    assert np.all(np.abs(first - x_t.numpy()[None]).max(axis=(1, 2, 3)) > 0)
    assert not np.allclose(first[0], first[1])


def test_same_seed_same_result_different_seed_differs(teacher, x_t):
    a = teacher.sample_transition(x_t, 0.6, 0.5, seeds=(11, 12))
    b = teacher.sample_transition(x_t, 0.6, 0.5, seeds=(11, 12))
    c = teacher.sample_transition(x_t, 0.6, 0.5, seeds=(13, 12))
    assert torch.equal(a.x_s, b.x_s)
    assert not torch.equal(a.x_s[0], c.x_s[0])
    assert torch.equal(a.x_s[1], c.x_s[1])       # per-rollout streams: seed 12 unaffected by its neighbour
    assert a.seeds == (11, 12)


def test_deterministic_control_repeats_exactly(teacher, x_t):
    a = teacher.ode_transition(x_t, 0.6, 0.5, n_copies=2)
    b = teacher.ode_transition(x_t, 0.6, 0.5, n_copies=2)
    assert torch.equal(a.x_s, b.x_s)
    assert torch.equal(a.x_s[0], a.x_s[1])
    assert a.method["probability_flow"] and a.method["corrector"] == "none"
    s = teacher.sample_transition(x_t, 0.6, 0.5, seeds=(0,))
    assert not torch.allclose(a.x_s[0], s.x_s[0])


def test_output_shapes_dtypes_devices_and_nfe(teacher, x_t):
    r = teacher.sample_transition(x_t, 0.6, 0.55, seeds=range(4))
    assert r.x_s.shape == (4, 2, 1, 1) and r.x_s.dtype == torch.float32
    assert r.x_s.device.type == "cpu"
    assert r.n_steps == r.i_s - r.i_t > 0
    assert r.nfe_per_sample == r.n_steps                         # predictor only: 1 NFE per step
    pc = teacher.sample_transition(x_t, 0.6, 0.55, seeds=range(4), corrector="langevin", n_steps_each=1)
    assert pc.nfe_per_sample == 2 * pc.n_steps                   # + one Langevin step per level
    h = (teacher.sde.T - teacher.eps) / (teacher.sde.N - 1)     # grid spacing: times snap to it
    assert r.t == pytest.approx(0.6, abs=h / 2) and r.s == pytest.approx(0.55, abs=h / 2)


def test_record_at_snapshot_equals_the_shorter_transition(teacher, x_t):
    i_t, i_mid, _, s_mid = teacher.transition_indices(0.6, 0.55)
    long = teacher.sample_transition(x_t, 0.6, 0.5, seeds=(5, 6), record_at=[i_mid])
    short = teacher.sample_transition(x_t, 0.6, s_mid, seeds=(5, 6))
    assert long.intermediate_steps.tolist() == [i_mid]
    assert np.array_equal(long.intermediates[0], short.x_s.numpy())


def test_reverse_diffusion_matches_exact_affine_kernel(teacher, x_t):
    B = 4000
    r = teacher.sample_transition(x_t, 0.6, 0.45, seeds=range(B))
    X = r.x_s.numpy().reshape(B, 2).astype(np.float64)
    m, Cv = exact_kernel(teacher, x_t.numpy(), r.i_t, r.i_s)
    se = np.sqrt(np.diag(Cv) / B)
    assert np.all(np.abs(X.mean(0) - m) < 4 * se), (X.mean(0), m, se)
    Ce = np.cov(X, rowvar=False)
    assert np.allclose(Ce, Cv, rtol=0.12, atol=0.12 * np.abs(Cv).max()), (Ce, Cv)


def test_discretised_kernel_approaches_the_bayes_posterior(x_t):
    """As the grid is refined the SMLD kernel converges to the exact time-reversal conditional."""
    errs = []
    for N in (400, 3200):
        T = make_teacher(N=N)
        i_t, i_s, tg, sg = T.transition_indices(0.6, 0.45)
        m, Cv = exact_kernel(T, x_t.numpy(), i_t, i_s)
        mb, Cb = bayes_posterior(x_t.numpy(), _sigma(tg), _sigma(sg))
        errs.append((np.abs(m - mb).max(), np.abs(Cv - Cb).max()))
    assert errs[1][0] < errs[0][0] and errs[1][1] < errs[0][1]
    assert errs[1][0] < 0.05 * np.abs(mb).max() and errs[1][1] < 0.05 * np.abs(Cb).max()


def test_tweedie_denoise_is_the_gaussian_posterior_mean(teacher):
    x0 = torch.tensor([1.3, -0.4], dtype=torch.float32).view(2, 1, 1)
    xt, info = teacher.forward_noise(x0, 0.5, seed=3)
    got = teacher.tweedie_denoise(xt, info["t"])[0].numpy().reshape(2)
    s2 = info["sigma"] ** 2
    want = MU + SIGMA @ np.linalg.solve(SIGMA + s2 * np.eye(2), xt.numpy().reshape(2) - MU)
    assert np.allclose(got, want, atol=1e-4)
