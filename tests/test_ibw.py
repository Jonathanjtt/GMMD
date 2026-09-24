"""IBW / MD / NGD on analytic low-dimensional targets whose answer is known in closed form.

These pin the algorithm itself: the family, the Prop. 3.1 gradients, each of the three variance
geometries, the positivity properties the paper claims for them, and the fixed points.  The
separate ``test_ibw_reference_parity.py`` checks the same code against the authors' reference
implementation.
"""

import numpy as np
import pytest

from gmmd import ibw
from gmmd.vi import TeacherConditionalTarget, isotropic_mixture_target


# ======================================================================================
# the variational family
# ======================================================================================
def test_mixture_density_and_score_agree_with_finite_differences():
    q = ibw.IsotropicMixture([[1.0, 0.0], [-2.0, 1.5], [0.0, 3.0]], [0.5, 1.0, 2.0])
    x = np.array([[0.3, 0.2], [-1.0, 2.0], [5.0, -4.0]])
    got = q.score(x)
    eps, fd = 1e-5, np.empty_like(x)
    for k in range(x.shape[1]):
        h = np.zeros_like(x)
        h[:, k] = eps
        fd[:, k] = (q.log_prob(x + h) - q.log_prob(x - h)) / (2 * eps)
    assert np.allclose(got, fd, atol=1e-6)
    # a proper density: integrates to 1 on a grid wide enough to contain it
    g = np.linspace(-12, 12, 400)
    X, Y = np.meshgrid(g, g)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    mass = np.exp(q.log_prob(pts)).sum() * (g[1] - g[0]) ** 2
    assert mass == pytest.approx(1.0, abs=1e-3)


def test_score_is_stable_far_from_every_component():
    q = ibw.IsotropicMixture([[0.0], [10.0]], [1e-3, 1e-3])
    s = q.score(np.array([[500.0]]))                 # every component log-density underflows
    assert np.all(np.isfinite(s))
    assert s[0, 0] < 0                               # still points back toward the mixture


def test_family_invariants():
    q = ibw.IsotropicMixture([[0.0, 0.0], [1.0, 1.0]], [1.0, 2.0])
    assert q.n_components == 2 and q.dim == 2
    assert np.allclose(q.weights, 0.5)               # uniform by construction
    assert q.covariances.shape == (2, 2, 2)
    assert np.allclose(q.covariances[1], 2.0 * np.eye(2))
    with pytest.raises(ValueError):
        ibw.IsotropicMixture([[0.0, 0.0]], [-1.0])   # variances must be positive
    with pytest.raises(ValueError):
        ibw.IsotropicMixture([[0.0, 0.0]], [1.0, 2.0])


def test_shared_noise_reaches_every_component():
    q = ibw.IsotropicMixture([[0.0, 0.0], [5.0, 5.0]], [1.0, 4.0])
    z = np.array([[1.0, -1.0], [0.0, 2.0]])
    s = q.sample_per_component(z)
    assert s.shape == (2, 2, 2)
    assert np.allclose(s[0], q.means[0] + 1.0 * z)   # sqrt(1)
    assert np.allclose(s[1], q.means[1] + 2.0 * z)   # sqrt(4)


# ======================================================================================
# gradients -- Proposition 3.1
# ======================================================================================
def test_gradients_are_exactly_the_reparametrised_gradient_of_the_estimator():
    """Prop. 3.1 is the derivative of E_{x~q_j}[log(q/pi)(x)] with the DENSITY q held fixed.

    Differentiating the full Monte-Carlo KL instead would add a second term -- the dependence
    of log q on its own parameters -- whose expectation under q is zero but whose finite-sample
    value is not.  Prop. 3.1 omits that term by construction, so the exact statement to check is
    a finite difference in which only the sample POSITIONS move.
    """
    rng = np.random.default_rng(0)
    target = isotropic_mixture_target([[3.0, 0.0], [-3.0, 0.0]], [1.0, 1.0])
    means0 = np.array([[1.0, 0.5], [-0.5, -1.0]])
    var0 = np.array([0.7, 1.3])
    noise = rng.standard_normal((512, 2))
    frozen = ibw.IsotropicMixture(means0, var0)          # the density, held fixed throughout

    def F_frozen(means, variances):
        s = ibw.IsotropicMixture(means, variances).sample_per_component(noise)
        flat = s.reshape(-1, 2)
        return float(np.mean(frozen.log_prob(flat) - target.log_prob(flat)))

    gm, gv = ibw.mixture_gradients(frozen, target, noise)
    h = 1e-6
    for j in range(2):
        for k in range(2):
            mp, mm = means0.copy(), means0.copy()
            mp[j, k] += h
            mm[j, k] -= h
            assert gm[j, k] == pytest.approx((F_frozen(mp, var0) - F_frozen(mm, var0)) / (2 * h),
                                             abs=1e-7)
        vp, vm = var0.copy(), var0.copy()
        vp[j] += h
        vm[j] -= h
        assert gv[j] == pytest.approx((F_frozen(means0, vp) - F_frozen(means0, vm)) / (2 * h),
                                      abs=1e-7)


def test_gradients_target_the_true_kl_gradient_where_it_is_known_in_closed_form():
    """N = 1 against a single Gaussian: KL and both its gradients have closed forms.

    KL(N(m, vI) | N(mu, sI)) = 0.5 [ d v/s + |m-mu|^2/s - d + d log(s/v) ], so
    grad_m = (m - mu)/s  and  grad_v = 0.5 (d/s - d/v).  This is what pins the estimator to the
    right objective; the frozen-density test above only pins its algebra.
    """
    d, mu, s = 2, np.array([2.0, -1.0]), 0.8
    m, v = np.array([[0.5, 0.5]]), 1.7
    target = isotropic_mixture_target(mu[None], [s])
    q = ibw.IsotropicMixture(m, [v])
    noise = np.random.default_rng(3).standard_normal((200_000, d))
    gm, gv = ibw.mixture_gradients(q, target, noise)
    assert np.allclose(gm[0], (m[0] - mu) / s, atol=5e-3)
    assert gv[0] == pytest.approx(0.5 * (d / s - d / v), abs=5e-3)


def test_gradients_vanish_when_q_equals_pi():
    """q = pi is a stationary point of the KL, so both gradients are zero (any noise block)."""
    means, var = np.array([[2.0, 0.0], [-2.0, 0.0]]), np.array([0.6, 0.6])
    target = isotropic_mixture_target(means, var)
    q = ibw.IsotropicMixture(means, var)
    gm, gv = ibw.mixture_gradients(q, target, np.random.default_rng(1).standard_normal((20000, 2)))
    assert np.abs(gm).max() < 5e-3
    assert np.abs(gv).max() < 5e-3


# ======================================================================================
# the three variance geometries
# ======================================================================================
def test_variance_updates_are_the_paper_formulas():
    var, g, N, d, step = np.array([0.5, 2.0]), np.array([0.3, -0.8]), 4, 3, 0.05
    u = (2.0 * N * step / d) * g
    assert np.allclose(ibw.ibw_variance_update(var, g, N, d, step), (1 - u) ** 2 * var)
    assert np.allclose(ibw.md_variance_update(var, g, N, d, step), var * np.exp(-u))
    m = np.zeros((2, 3))
    nm, nv, ok = ibw.ngd_update(m, var, np.ones((2, 3)), g, N, d, step)
    assert ok and np.allclose(nv, 1.0 / (1.0 / var + u))
    assert np.allclose(nm, m - step * N * nv[:, None] * np.ones((2, 3)))


@pytest.mark.parametrize("g", [-50.0, -1.0, 0.0, 1.0, 50.0])
def test_ibw_and_md_keep_variances_positive_for_any_gradient(g):
    var = np.array([0.5, 2.0])
    grad = np.array([g, -g])
    assert np.all(ibw.ibw_variance_update(var, grad, 5, 3, 0.1) >= 0)
    assert np.all(ibw.md_variance_update(var, grad, 5, 3, 0.1) > 0)


def test_ngd_reports_loss_of_positivity_instead_of_clipping():
    """The paper states Eq. (14) carries no positivity guarantee; it must be surfaced."""
    var = np.array([1.0])
    _, _, ok = ibw.ngd_update(np.zeros((1, 2)), var, np.zeros((1, 2)), np.array([-100.0]),
                              n_components=1, dim=2, step=1.0)
    assert ok is False


# ======================================================================================
# Algorithm 1 end to end
# ======================================================================================
def test_single_gaussian_target_is_recovered_exactly():
    m, v = np.array([2.0, -1.0]), 0.3
    target = isotropic_mixture_target(m[None], [v])
    r = ibw.fit(target, n_components=1, method="ibw", n_iterations=800, step_size=0.05,
                n_grad_samples=64, init_means=np.zeros((1, 2)), init_variances=[1.0], seed=0)
    assert r["stopped_early"] is None
    assert np.allclose(r["means"][0], m, atol=0.05)
    assert r["variances"][0] == pytest.approx(v, rel=0.1)
    assert r["kls"][-1] < r["kls"][0]


@pytest.mark.parametrize("method", ["ibw", "md"])
def test_two_modes_get_one_component_each(method):
    modes = np.array([[3.0, 0.0], [-3.0, 0.0]])
    target = isotropic_mixture_target(modes, [0.5, 0.5])
    r = ibw.fit(target, n_components=2, method=method, n_iterations=900, step_size=0.05,
                n_grad_samples=64, init_means=np.array([[1.0, 0.5], [-1.0, -0.5]]),
                init_variances=[1.0, 1.0], seed=0)
    assert r["stopped_early"] is None
    d = np.linalg.norm(r["means"][:, None, :] - modes[None, :, :], axis=-1)
    assert set(d.argmin(axis=1)) == {0, 1}                 # one component per mode
    assert d.min(axis=1).max() < 0.15
    assert np.allclose(r["variances"], 0.5, rtol=0.2)
    assert np.allclose(r["weights"], 0.5)


def test_fit_is_reproducible_and_seed_sensitive():
    target = isotropic_mixture_target([[2.0, 0.0], [-2.0, 0.0]], [1.0, 1.0])
    kw = dict(n_components=3, dim=2, n_iterations=60, step_size=0.02, n_grad_samples=8)
    a = ibw.fit(target, seed=0, **kw)
    b = ibw.fit(target, seed=0, **kw)
    c = ibw.fit(target, seed=1, **kw)
    assert np.array_equal(a["means"], b["means"]) and np.array_equal(a["variances"], b["variances"])
    assert not np.allclose(a["means"], c["means"])


def test_traces_shapes_and_the_initial_snapshot():
    target = isotropic_mixture_target([[0.0, 0.0]], [1.0])
    init = np.array([[1.0, 1.0], [-1.0, 2.0]])
    r = ibw.fit(target, method="ibw", n_iterations=25, step_size=0.01, n_grad_samples=8,
                init_means=init, init_variances=[1.0, 1.0], seed=0, kl_every=5)
    assert r["means_trace"].shape == (25, 2, 2) and r["variances_trace"].shape == (25, 2)
    assert np.array_equal(r["means_trace"][0], init)       # index 0 is the initial mixture
    assert r["kls"].shape == r["kl_iters"].shape and r["kl_iters"][0] == 0
    assert r["settings"]["method"] == "ibw" and r["n_iterations_run"] == 25


def test_common_noise_reuses_one_block():
    target = isotropic_mixture_target([[2.0, 0.0]], [1.0])
    kw = dict(n_components=2, dim=2, n_iterations=40, step_size=0.02, n_grad_samples=8, seed=0)
    assert not np.allclose(ibw.fit(target, common_noise=True, **kw)["means"],
                           ibw.fit(target, common_noise=False, **kw)["means"])


def test_unknown_method_and_missing_dim_are_refused():
    target = isotropic_mixture_target([[0.0, 0.0]], [1.0])
    with pytest.raises(ValueError, match="method must be one of"):
        ibw.fit(target, method="nope", dim=2, n_iterations=1)
    bare = type("T", (), {"score": staticmethod(lambda x: -x),
                          "log_prob": staticmethod(lambda x: -(x ** 2).sum(1) / 2)})()
    with pytest.raises(ValueError, match="carries no dimension"):
        ibw.fit(bare, n_iterations=1)


def test_teacher_conditional_target_is_blocked():
    with pytest.raises(NotImplementedError, match=r"grad_\{x_s\} log p_theta"):
        TeacherConditionalTarget(None, None, 0.6, 0.45)
