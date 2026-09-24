"""The torch backend reproduces the NumPy reference implementation of IBW.

:mod:`gmmd.ibw` is the readable reference, itself checked against the paper authors' own code in
``test_ibw_reference_parity.py``.  :mod:`gmmd.ibw_gpu` is the fast path used when the target
score is a neural network.  This file is the link between them: if it passes, the GPU results
inherit the reference's provenance.

Agreement is to float32 tolerance, not machine precision, because the GPU backend runs in
float32 by design -- that is the point of it.  A float64 run of the torch backend is also
checked, and there the two must agree far more tightly.
"""

import numpy as np
import pytest

from gmmd import ibw
from gmmd.vi import isotropic_mixture_target

torch = pytest.importorskip("torch")


class _TorchTarget:
    """An analytic isotropic-mixture target exposing BOTH interfaces on the same parameters."""

    def __init__(self, means, variances, weights=None, device="cpu", dtype=torch.float64):
        self._np = isotropic_mixture_target(means, variances, weights)
        self.dim = self._np.dim
        self.device = device
        m = torch.as_tensor(np.asarray(means), dtype=dtype, device=device)
        v = torch.as_tensor(np.asarray(variances), dtype=dtype, device=device).reshape(-1)
        w = torch.full((len(v),), 1.0 / len(v), dtype=dtype, device=device) if weights is None \
            else torch.as_tensor(np.asarray(weights), dtype=dtype, device=device)
        self._m, self._v, self._logw = m, v, torch.log(w / w.sum())

    def score(self, x):
        return self._np.score(x)

    def score_torch(self, x):
        d = self._m.shape[1]
        quad = torch.stack([((x - self._m[k]) ** 2).sum(1) / self._v[k]
                            for k in range(len(self._v))], dim=1)
        logc = (self._logw - 0.5 * d * (np.log(2 * np.pi) + torch.log(self._v)))[None, :] - 0.5 * quad
        r = torch.softmax(logc, dim=1)
        out = torch.zeros_like(x)
        for k in range(len(self._v)):
            out -= (r[:, k] / self._v[k])[:, None] * (x - self._m[k][None, :])
        return out


MEANS = np.array([[3.0, 0.0, 1.0], [-3.0, 1.0, 0.0], [0.0, 4.0, -2.0]])
VARS = np.array([0.5, 1.2, 0.8])
INIT_M = np.array([[1.0, 0.5, 0.0], [-0.5, -1.0, 0.5], [2.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
INIT_V = np.array([0.7, 1.3, 0.4, 2.0])


def test_mixture_density_and_score_match_the_reference():
    from gmmd.ibw_gpu import TorchIsotropicMixture

    qn = ibw.IsotropicMixture(INIT_M, INIT_V)
    qt = TorchIsotropicMixture(INIT_M, INIT_V, device="cpu", dtype=torch.float64)
    x = np.random.default_rng(0).standard_normal((40, 3)) * 2
    xt = torch.as_tensor(x, dtype=torch.float64)
    assert np.abs(qn.log_prob(x) - qt.log_prob(xt).numpy()).max() < 1e-10
    assert np.abs(qn.score(x) - qt.score(xt).numpy()).max() < 1e-10
    assert qt.n_components == 4 and qt.dim == 3
    assert np.allclose(qt.weights.numpy(), 0.25)


def test_gradients_match_the_reference_on_a_shared_noise_block():
    from gmmd import ibw_gpu

    tgt = _TorchTarget(MEANS, VARS)
    noise = np.random.default_rng(1).standard_normal((16, 3))
    gm_n, gv_n = ibw.mixture_gradients(ibw.IsotropicMixture(INIT_M, INIT_V), tgt, noise)
    gm_t, gv_t = ibw_gpu.mixture_gradients(
        ibw_gpu.TorchIsotropicMixture(INIT_M, INIT_V, dtype=torch.float64), tgt,
        torch.as_tensor(noise, dtype=torch.float64))
    assert np.abs(gm_n - gm_t.numpy()).max() < 1e-10
    assert np.abs(gv_n - gv_t.numpy()).max() < 1e-10


@pytest.mark.parametrize("method", ["ibw", "md"])
def test_full_fit_matches_the_reference_in_float64(method):
    """Same init, same noise block, same update rule: the trajectories must coincide exactly.

    Both backends are handed ONE explicit noise block reused every iteration, which removes the
    only source of divergence between them (their RNG streams) and makes this a test of the
    arithmetic rather than of Monte-Carlo agreement.
    """
    from gmmd import ibw_gpu

    tgt = _TorchTarget(MEANS, VARS)
    n_iter, B = 60, 12
    noise = np.random.default_rng(3).standard_normal((B, 3))
    kw = dict(n_components=4, method=method, n_iterations=n_iter, step_size=0.05,
              n_grad_samples=B, init_means=INIT_M, init_variances=INIT_V, seed=0,
              fixed_noise=noise)

    ref = ibw.fit(tgt, kl_every=0, **kw)
    got = ibw_gpu.fit(tgt, device="cpu", dtype=torch.float64, **kw)

    assert ref["stopped_early"] is None and got["stopped_early"] is None
    assert np.abs(got["means"] - ref["means"]).max() < 1e-9
    assert np.abs(got["variances"] - ref["variances"]).max() < 1e-9
    assert np.abs(got["variances_trace"] - ref["variances_trace"]).max() < 1e-9


def test_float32_backend_stays_close_to_the_float64_reference(method="ibw"):
    """The backend runs in float32 by design; that must cost accuracy, not correctness."""
    from gmmd import ibw_gpu

    tgt32 = _TorchTarget(MEANS, VARS, dtype=torch.float32)
    tgt64 = _TorchTarget(MEANS, VARS)
    noise = np.random.default_rng(5).standard_normal((12, 3))
    kw = dict(n_components=4, method=method, n_iterations=60, step_size=0.05, n_grad_samples=12,
              init_means=INIT_M, init_variances=INIT_V, seed=0, fixed_noise=noise)
    ref = ibw.fit(tgt64, kl_every=0, **kw)
    got = ibw_gpu.fit(tgt32, device="cpu", dtype=torch.float32, **kw)
    assert np.abs(got["means"] - ref["means"]).max() < 1e-3
    assert np.abs(got["variances"] - ref["variances"]).max() < 1e-3


def test_gpu_backend_refuses_ngd():
    from gmmd import ibw_gpu

    with pytest.raises(ValueError, match="implements 'ibw' and 'md'"):
        ibw_gpu.fit(_TorchTarget(MEANS, VARS), n_components=2, method="ngd",
                    init_means=INIT_M[:2], init_variances=INIT_V[:2])


def test_fit_is_reproducible_and_seed_sensitive():
    from gmmd import ibw_gpu

    tgt = _TorchTarget(MEANS, VARS)
    kw = dict(n_components=3, n_iterations=25, step_size=0.05, n_grad_samples=8,
              init_means=INIT_M[:3], init_variances=INIT_V[:3], device="cpu",
              dtype=torch.float64)
    a = ibw_gpu.fit(tgt, seed=0, **kw)
    b = ibw_gpu.fit(tgt, seed=0, **kw)
    c = ibw_gpu.fit(tgt, seed=1, **kw)
    assert np.array_equal(a["means"], b["means"])
    assert not np.allclose(a["means"], c["means"])
    assert a["grad_norms"].shape == (25,) and np.all(np.isfinite(a["grad_norms"]))
