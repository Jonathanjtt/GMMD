"""The single-interval replacement plumbing, on the analytic toy teacher (CPU, no checkpoint).

* splitting the official loop at ``i_t < i_s`` reproduces the uninterrupted loop bit for bit;
* the coarse sampler with one fine step per block is the fine sampler;
* each image's trajectory is independent of the batch it ran in;
* the batched IBW fitter reproduces ``gmmd.ibw_gpu.fit`` anchor by anchor;
* mixture sampling draws random points, not the mean.
"""

import numpy as np
import pytest
import torch

from gmmd import ibw_batched, ibw_gpu
from gmmd import pipeline as P
from test_ibw_gpu_parity import _TorchTarget
from test_teacher_toy import make_teacher


def test_split_equals_uninterrupted():
    T = make_teacher()
    seeds = [3, 17, 40]
    N, i_t, i_s = T.sde.N, 150, 170
    x0 = P.prior_sample(T, seeds)

    # split run: prefix / segment / suffix with their own stage generators
    x, _, n1 = P.fine_steps(T, x0, 0, i_t, P.Generators(seeds, "prefix", "cpu"))
    x, _, n2 = P.fine_steps(T, x, i_t, i_s, P.Generators(seeds, "segment", "cpu"))
    x, xm_split, n3 = P.fine_steps(T, x, i_s, N, P.Generators(seeds, "suffix", "cpu"))
    assert n1 + n2 + n3 == len(seeds) * N

    # uninterrupted run drawing from the same three streams at the same steps
    from gmmd.teachers.score_sde._vendor import sampling
    pred = sampling.get_predictor("reverse_diffusion")(T.sde, T._score_fn, False)
    g = {k: P.Generators(seeds, k, "cpu") for k in ("prefix", "segment", "suffix")}
    x = x0
    with torch.no_grad():
        for i in range(N):
            stage = "prefix" if i < i_t else "segment" if i < i_s else "suffix"
            with T._noise_from(g[stage].gens):
                x, xm = pred.update_fn(x, torch.ones(len(seeds)) * T.timesteps[i])
    assert torch.equal(xm, xm_split)


def test_batch_independence():
    T = make_teacher()
    a, _, _ = P.fine_steps(T, P.prior_sample(T, [5, 6, 7]), 0, 60,
                           P.Generators([5, 6, 7], "prefix", "cpu"), batch_size=2)
    b, _, _ = P.fine_steps(T, P.prior_sample(T, [6]), 0, 60, P.Generators([6], "prefix", "cpu"))
    assert torch.equal(a[1:2], b)


def test_coarse_with_unit_blocks_is_fine():
    T = make_teacher()
    seeds = [1, 2]
    x0, _, _ = P.fine_steps(T, P.prior_sample(T, seeds), 0, 100, P.Generators(seeds, "prefix", "cpu"))
    xf, mf, _ = P.fine_steps(T, x0, 100, 120, P.Generators(seeds, "segment", "cpu"))
    xc, mc, nfe = P.coarse_steps(T, x0, 100, 120, 20, P.Generators(seeds, "segment", "cpu"))
    assert nfe == 20 * len(seeds)
    torch.testing.assert_close(xc, xf, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mc, mf, rtol=1e-5, atol=1e-5)


def test_coarse_blocks_cover_interval_once():
    blocks = P.coarse_blocks(800, 840, 4)
    assert blocks[0][0] == 800 and blocks[-1][1] == 840
    assert all(b0 == a1 for (_, a1), (b0, _) in zip(blocks[:-1], blocks[1:]))
    with pytest.raises(ValueError):
        P.coarse_blocks(800, 802, 4)


def test_coarse_noise_level_matches_fine():
    """The coarse step lands on the same noise level as the fine steps it replaces."""
    T = make_teacher()
    (a, b), = P.coarse_blocks(100, 120, 1)
    sig_a, _ = P.step_sigmas(T, a)
    _, adj = P.step_sigmas(T, b - 1)
    total = sum(P.step_sigmas(T, i)[0] ** 2 - P.step_sigmas(T, i)[1] ** 2 for i in range(a, b))
    assert abs((sig_a ** 2 - adj ** 2) - total) < 1e-3 * total


@pytest.mark.parametrize("K", [1, 2])
def test_batched_ibw_matches_single(K):
    means = np.array([[3.0, 0.0, 1.0], [-3.0, 1.0, 0.0]])
    tgt = _TorchTarget(means, [1.0, 0.5], dtype=torch.float32)
    rng = np.random.default_rng(0)
    A, d = 3, 3
    init_m = rng.standard_normal((A, K, d))
    init_v = np.full((A, K), 2.0)
    seeds = [11, 12, 13]

    def score_fn(x, anchor):
        return tgt.score_torch(x)

    res = ibw_batched.fit_batched(score_fn, init_m, init_v, seeds, n_iterations=40,
                                  step_size=0.05, n_grad_samples=4, device="cpu")
    for a in range(A):
        ref = ibw_gpu.fit(tgt, n_components=K, n_iterations=40, step_size=0.05,
                          n_grad_samples=4, init_means=init_m[a], init_variances=init_v[a],
                          seed=seeds[a], device="cpu")
        np.testing.assert_allclose(res["means"][a].numpy(), ref["means"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(res["variances"][a].numpy(), ref["variances"], rtol=1e-4)
        np.testing.assert_allclose(res["grad_norms"][:, a], ref["grad_norms"], rtol=1e-4)


def test_mixture_sample_is_random():
    m = torch.zeros(2, 1, 5)
    v = torch.ones(2, 1)
    gens = [torch.Generator().manual_seed(s) for s in (0, 1)]
    x = ibw_batched.sample_batched(m, v, 3, gens)
    assert x.shape == (2, 3, 5)
    assert x.abs().mean() > 0.3                      # not the mean
    assert not torch.equal(x[0], x[1])                # anchors draw independently
