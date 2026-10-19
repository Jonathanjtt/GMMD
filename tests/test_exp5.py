"""exp5 plumbing on the analytic toy teacher (CPU, no checkpoint), plus one GPU checkpoint test.

* the partition covers the grid contiguously, ends on the sigma_min state + noise removal;
* recording interval starts reproduces the uninterrupted sampler bit for bit;
* interval rollouts are batch-independent and the terminal one returns the noise-removed mean;
* the coarse full-trajectory sampler with unit blocks is the fine sampler;
* the principal-direction initialisation recovers the exact Gaussian posterior covariance;
* the mixture likelihood sums each component over all coordinates before mixing, and the
  standardised NLL is the plain NLL shifted by a constant;
* IBW convergence diagnostics do not change the iterates;
* (checkpoint) the student starts at the teacher's Tweedie mean and the standalone sampler makes
  exactly 10 network evaluations and 10 GMM draws per image, with zero teacher/IBW calls.
"""

import numpy as np
import pytest
import torch

from gmmd import ibw_batched
from gmmd import pipeline as P
from gmmd.exp5 import grid as G
from gmmd.exp5 import ibw_targets as IT
from gmmd.exp5.student import mixture_log_prob, standardized_nll
from test_teacher_toy import MU, SIGMA, make_teacher


@pytest.fixture(scope="module")
def teacher():
    return make_teacher()


@pytest.fixture(scope="module")
def parts(teacher):
    return G.make_partition(teacher, 10)


def test_partition_covers_grid(teacher, parts):
    N = teacher.sde.N
    assert parts[0].step_start == 0 and parts[-1].step_end == N
    assert all(a.step_end == b.step_start for a, b in zip(parts[:-1], parts[1:]))
    assert sum(iv.n_steps for iv in parts) == N and {iv.n_steps for iv in parts} == {N // 10}
    assert [iv.terminal for iv in parts] == [False] * 9 + [True]
    assert parts[-1].grid_end_state == N - 1 and parts[-1].discrete_sigma_end == 0.0
    for iv in parts:
        assert iv.gap > 0 and 0 < iv.v_anc < iv.gap
        assert iv.mean_gap == pytest.approx(iv.sigma_start ** 2 if iv.terminal else iv.gap)
    # each interval ends where the next begins (state times and noise levels)
    for a, b in zip(parts[:-1], parts[1:]):
        assert a.t_end_state == b.t_start and a.sigma_end_state == pytest.approx(b.sigma_start)


def test_trajectory_states_match_uninterrupted(teacher, parts):
    seeds = [7, 8, 9]
    states, nfe = G.trajectory_states(teacher, seeds, parts)
    assert nfe == len(seeds) * parts[-1].step_start
    gens = G.SeededGenerators([G.seed_of(51, s) for s in seeds], "cpu")
    x = P.prior_sample(teacher, seeds)
    from gmmd.teachers.score_sde._vendor import sampling
    pred = sampling.get_predictor("reverse_diffusion")(teacher.sde, teacher._score_fn, False)
    starts = {iv.step_start: iv.j for iv in parts}
    with torch.no_grad(), teacher._noise_from(gens.gens):
        for i in range(parts[-1].step_start + 1):
            if i in starts:
                assert torch.equal(states[starts[i]], x)
            if i < parts[-1].step_start:
                x, _ = pred.update_fn(x, torch.ones(len(seeds)) * teacher.timesteps[i])


def test_interval_rollouts_batch_independent_and_terminal_mean(teacher, parts):
    x = torch.randn(2, 2, 1, 1) * parts[9].sigma_start + torch.tensor(MU, dtype=torch.float32).view(1, 2, 1, 1)
    seeds = np.array([[1, 2, 3], [4, 5, 6]])
    y, nfe = G.interval_rollouts(teacher, x, parts[9], seeds)
    assert y.shape == (2, 3, 2, 1, 1) and nfe == 6 * parts[9].n_steps
    y1, _ = G.interval_rollouts(teacher, x[1:], parts[9], seeds[1:, 1:2])
    assert torch.equal(y[1, 1], y1[0, 0])
    # terminal: rolling the stochastic part, then the deterministic terminal map, gives the same output
    gens = G.SeededGenerators([4], "cpu")
    xs, _, _ = P.fine_steps(teacher, x[1:], parts[9].step_start, teacher.sde.N - 1, gens)
    x0, n = G.terminal_denoise(teacher, xs)
    assert n == 1
    torch.testing.assert_close(x0, y[1, 0:1], rtol=1e-6, atol=1e-6)


def test_coarse_unit_blocks_is_fine(teacher):
    seeds = [3, 4]
    ns = [11, 12]
    x0c, nfe = G.coarse_trajectory(teacher, seeds, teacher.sde.N, kind="reverse_diffusion", noise_seeds=ns)
    assert nfe == len(seeds) * teacher.sde.N
    _, x0f, _ = P.fine_steps(teacher, P.prior_sample(teacher, seeds), 0, teacher.sde.N, G.SeededGenerators(ns, "cpu"))
    torch.testing.assert_close(x0c, x0f, rtol=1e-4, atol=1e-4)


def test_coarse_ten_steps_counts_and_is_finite(teacher):
    for kind in G.COARSE_KINDS:
        x0, nfe = G.coarse_trajectory(teacher, [1, 2, 3], 10, kind=kind)
        assert nfe == 30 and torch.isfinite(x0).all()


def test_principal_directions_recover_gaussian_posterior(teacher, parts):
    iv = parts[5]
    x = torch.tensor([[0.3, -2.0]], dtype=torch.float32).view(1, 2, 1, 1)
    with IT.tf32(False):
        lam, E, nfe = IT.posterior_principal_directions(teacher, iv, x, q=2, n_iter=6, oversample=0, rel_h=0.05)
    Ss = SIGMA + iv.sigma_end_state ** 2 * np.eye(2)
    cov = np.linalg.inv(np.linalg.inv(Ss) + np.eye(2) / iv.gap)
    w, V = np.linalg.eigh(cov)
    np.testing.assert_allclose(lam[0].numpy(), w[::-1], rtol=2e-3)
    for k in range(2):
        assert abs(abs(float(E[0, k].numpy() @ V[:, ::-1][:, k])) - 1) < 1e-3
    assert nfe == 2 * 2 * 7
    m, v, info = IT.initial_mixture("principal", teacher, iv, x, 4, [0], spread=2.0, var="v_anc", principal=(lam, E))
    mu, _ = IT.tweedie_mean(teacher, iv, x)
    torch.testing.assert_close(m[0, 0] - mu[0], 2.0 * lam[0, 0].sqrt() * E[0, 0])
    torch.testing.assert_close(m[0, 1] - mu[0], -2.0 * lam[0, 0].sqrt() * E[0, 0])
    assert torch.all(v == iv.v_anc)


def test_principal_refuses_tf32(teacher, parts):
    with IT.tf32(True):
        if torch.backends.cudnn.allow_tf32:
            with pytest.raises(RuntimeError):
                IT.posterior_principal_directions(teacher, parts[3], torch.zeros(1, 2, 1, 1), q=1, oversample=0)


def test_mixture_likelihood_sums_over_dims_then_mixes():
    from scipy.special import logsumexp
    from scipy.stats import norm
    g = torch.Generator().manual_seed(0)
    K, d, M = 3, 5, 4
    means = torch.randn(K, d, generator=g) * 3
    var = torch.tensor([0.5, 1.0, 2.0])
    z = torch.randn(M, d, generator=g) * 2
    logits = torch.tensor([0.1, -0.3, 0.7])
    ref = np.array([logsumexp([np.log(np.exp(logits.numpy()) / np.exp(logits.numpy()).sum())[k]
                               + norm.logpdf(z[m].numpy(), means[k].numpy(), np.sqrt(var[k].item())).sum()
                               for k in range(K)]) for m in range(M)])
    got = mixture_log_prob(z[None], means[None], var.log()[None], logits[None])[0]
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-5)
    iso = IT.isotropic_log_prob(z, means, var)                     # uniform weights
    got_u = mixture_log_prob(z[None], means[None], var.log()[None], torch.zeros(1, K))[0]
    np.testing.assert_allclose(got_u.numpy(), iso.numpy(), rtol=1e-6)
    # a point mixing component 0 in some coordinates with component 1 in the others is NOT a
    # high-likelihood point of the mixture (no per-pixel mixing)
    chimera = torch.cat([means[0, :2], means[1, 2:]])[None]
    assert mixture_log_prob(chimera[None], means[None], var.log()[None], logits[None])[0, 0] < \
        mixture_log_prob(means[:1][None], means[None], var.log()[None], logits[None])[0, 0] - 5


def test_standardized_nll_is_shifted_nll():
    g = torch.Generator().manual_seed(1)
    B, M, K, d = 2, 3, 4, 6
    x = torch.randn(B, d, generator=g) * 50
    means = x[:, None] + torch.randn(B, K, d, generator=g) * 7
    logv = torch.randn(B, K, generator=g) + 3
    logits = torch.randn(B, K, generator=g)
    z = x[:, None] + torch.randn(B, M, d, generator=g) * 7
    ls = torch.tensor([1.3, 2.1])
    s = standardized_nll(z, x, means, logv, logits, ls)
    plain = -mixture_log_prob(z, means, logv, logits) / d
    torch.testing.assert_close(s, plain - ls[:, None], rtol=1e-5, atol=1e-5)


def test_ibw_diagnostics_do_not_change_iterates():
    from test_ibw_gpu_parity import _TorchTarget
    tgt = _TorchTarget(np.array([[3.0, 0.0, 1.0], [-3.0, 1.0, 0.0]]), [1.0, 0.5], dtype=torch.float32)
    rng = np.random.default_rng(0)
    m0, v0 = rng.standard_normal((2, 3, 3)), np.full((2, 3), 2.0)
    f = lambda x, anchor: tgt.score_torch(x)
    c0 = ibw_batched.ITERATION_COUNTER[0]
    a = ibw_batched.fit_batched(f, m0, v0, [1, 2], n_iterations=20, step_size=0.05, device="cpu")
    b = ibw_batched.fit_batched(f, m0, v0, [1, 2], n_iterations=20, step_size=0.05, device="cpu", diag_every=5)
    assert ibw_batched.ITERATION_COUNTER[0] - c0 == 40
    assert torch.equal(a["means"], b["means"]) and torch.equal(a["variances"], b["variances"])
    dg = b["diagnostics"]
    assert list(dg["iterations"]) == [0, 5, 10, 15, 20] and dg["variances"].shape == (5, 2, 3)
    assert np.isnan(dg["mean_drift"][0]).all() and (dg["mean_drift"][1:] > 0).all()


def test_component_stats_merges_close_components():
    m = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
    s = IT.component_stats(m, torch.ones(3))
    assert s["n_effective"] == 2 and s["group_sizes"] == [2, 1]


# ---------------------------------------------------------------------------------- checkpoint
@pytest.mark.checkpoint
def test_student_init_and_standalone_sampler(tmp_path):
    from gmmd.datasets.celebahq import data_root
    from gmmd.download import CHECKPOINTS
    name = "ve/celebahq_256_ncsnpp_continuous"
    ckpt = data_root() / "checkpoints" / name / CHECKPOINTS[name]["file"]
    if not ckpt.exists() or not torch.cuda.is_available():
        pytest.skip("needs the checkpoint and a GPU")
    from gmmd.exp5.sampler import StudentSampler
    from gmmd.exp5.student import MixtureStudent
    from gmmd.teachers.score_sde import teacher as tm
    T = tm.ScoreSDETeacher.from_config(name, ckpt, device="cuda")
    parts = G.make_partition(T, 10)
    ivs = [dict(j=iv.j, sigma_start=iv.sigma_start, mean_gap=iv.mean_gap, log_vref=float(np.log(iv.v_anc)),
                v_anc=iv.v_anc, terminal=iv.terminal) for iv in parts]
    S = MixtureStudent.from_teacher(T.model, name, ivs, n_components=8).cuda().eval()
    x = torch.randn(2, 3, 256, 256, device="cuda") * parts[4].sigma_start + 0.5
    with torch.no_grad():
        logits, means, logv = S(x, 4)
        s = T.score(x, parts[4].t_start)
    torch.testing.assert_close(means[:, 0], x + parts[4].gap * s, rtol=1e-4, atol=1e-3)
    assert torch.all(logits == 0) and torch.allclose(logv, torch.full_like(logv, float(np.log(parts[4].v_anc))))
    assert (means[:, 1] - means[:, 0]).abs().max() > 0                # symmetry broken
    path = tmp_path / "ckpt.pt"
    torch.save(dict(arch=S.arch(), ema=S.state_dict(), step=0,
                    meta=dict(sigma_max=float(T.sde.sigma_max))), path)
    del T, S
    torch.cuda.empty_cache()
    t0, i0 = tm.SCORE_EXAMPLES[0], tm.INSTANCES[0]
    sampler = StudentSampler(path, device="cuda", precision="fp32")
    out, rec = sampler.sample([1_000_000, 1_000_001, 1_000_002], batch_size=2, keep_intermediates=True)
    assert rec["verified"], rec
    assert rec["network_evals_per_image"] == 10 and rec["gmm_draws_per_image"] == 10
    assert rec["teacher_score_evals"] == 0 and rec["ibw_iterations"] == 0
    assert tm.SCORE_EXAMPLES[0] == t0 and tm.INSTANCES[0] == i0
    assert out["images"].shape == (3, 3, 256, 256) and out["states"].shape == (3, 11, 3, 256, 256)
    # an image does not depend on its batch, and x_T is the teacher's prior draw
    out1, _ = sampler.sample([1_000_001], batch_size=1)
    torch.testing.assert_close(out1["images"][0], out["images"][1], rtol=1e-4, atol=1e-4)
    z = torch.randn((3, 256, 256), generator=torch.Generator().manual_seed(1_000_000)) * 348.0
    torch.testing.assert_close(out["states"][0, 0], z)


def test_claims_are_exclusive_and_stale_locks_reclaimed(tmp_path):
    import os
    from gmmd.exp5 import common as Cm
    lk = tmp_path / "locks" / "fit" / "x"
    assert Cm.claim(lk) and not Cm.claim(lk)           # second claimant is refused
    Cm.release(lk)
    assert Cm.claim(lk)
    host = (lk / "owner").read_text().split()[0]
    (lk / "owner").write_text(f"{host} 999999999")      # owner process no longer exists
    assert Cm.claim(lk) and (lk / "owner").read_text().split()[1] == str(os.getpid())
