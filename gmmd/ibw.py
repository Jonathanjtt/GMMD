"""Variational inference with a mixture of isotropic Gaussians -- IBW, MD and NGD.

Implements Algorithm 1 of

    Marguerite Petit-Talamon, Marc Lambert, Anna Korba,
    "Variational Inference with Mixtures of Isotropic Gaussians", NeurIPS 2025.
    https://arxiv.org/abs/2506.13613

for the variational family of *uniform-weight isotropic* Gaussian mixtures

    q(x) = (1/N) sum_j N(x; m_j, eps_j I),      m_j in R^d,  eps_j > 0,

minimising the reverse KL  F([m_j, eps_j]) = KL(q | pi)  (paper Eq. 7).  The target ``pi``
need not be normalised: only its score enters the updates.

Provenance.  This is an independent implementation from the paper's published equations,
NOT a copy of the authors' code.  Their reference implementation is
https://github.com/margueritetalamon/VI-MIG (``src/optim.py``, ``src/gmm.py``), which carries
no licence file, so its source is not redistributed here.  Every formula below cites the
paper equation it comes from, and ``tests/test_ibw_reference_parity.py`` clones VI-MIG at a
pinned commit *on demand* and asserts that the gradients and the one-step updates of this
module agree with theirs to machine precision on a fixed target, mixture and noise block.

The four equations, in the paper's notation (N components, dimension d, step size gamma):

    Prop 3.1   grad_{m_j} F   = (1/N) E_{x ~ N(m_j, eps_j I)}[ grad log(q/pi)(x) ]
               grad_{eps_j} F = (1/(2 N eps_j)) E[ (x - m_j)^T grad log(q/pi)(x) ]

    (GD)       m_j     <- m_j - gamma * N * grad_{m_j} F            (all three schemes)
    (12) IBW   eps_j   <- (1 - (2 N gamma / d) grad_{eps_j} F)^2 * eps_j
    (13) MD    eps_j   <- eps_j * exp( -(2 N gamma / d) grad_{eps_j} F )
    (14) NGD   1/eps_j <- 1/eps_j + (2 N gamma / d) grad_{eps_j} F
               m_j     <- m_j - gamma * N * eps_j^{new} * grad_{m_j} F   (NOT the plain (GD))

Positivity of ``eps`` is structural for IBW (a square) and for MD (an exponential); the paper
states explicitly that NGD carries no such guarantee, so :func:`fit` stops on the first
non-positive NGD variance and reports it rather than clipping the iterate.
"""

from __future__ import annotations

import numpy as np

__all__ = ["IsotropicMixture", "mixture_gradients", "ibw_variance_update",
           "md_variance_update", "ngd_update", "fit", "METHODS"]

METHODS = ("ibw", "md", "ngd")
# Above this many elements the per-iteration mean trace is not stored (see `fit`).
TRACE_MAX_ELEMENTS = 50_000_000
_LOG2PI = float(np.log(2.0 * np.pi))


# ======================================================================================
# the variational family
# ======================================================================================
class IsotropicMixture:
    """``q = (1/N) sum_j N(m_j, eps_j I)``: means ``(N, d)``, variances ``(N,)``, uniform weights.

    The weights are fixed at ``1/N`` by construction -- that is what makes the mixing measure
    ``(1/N) sum_j delta_(m_j, eps_j)`` identifiable (paper Sec. 4.2), and it is the family the
    paper's updates are derived for.  Nothing here optimises weights.
    """

    def __init__(self, means, variances):
        self.means = np.array(means, dtype=float, copy=True)
        if self.means.ndim != 2:
            raise ValueError(f"means must be (N, d), got {self.means.shape}")
        self.variances = np.array(variances, dtype=float, copy=True).reshape(-1)
        if self.variances.shape[0] != self.means.shape[0]:
            raise ValueError(f"{self.means.shape[0]} means but {self.variances.shape[0]} variances")
        if not np.all(self.variances > 0):
            raise ValueError("every variance must be strictly positive")

    @property
    def n_components(self):
        return self.means.shape[0]

    @property
    def dim(self):
        return self.means.shape[1]

    @property
    def weights(self):
        return np.full(self.n_components, 1.0 / self.n_components)

    # A dense (N, d, d) covariance is 309 GB per component at d = 196 608, so it is NOT built
    # unless asked for and the dimension is small.  `variances` is the complete parametrisation:
    # component j is eps_j * I, and every formula in this module uses the scalar.
    DENSE_COVARIANCE_MAX_DIM = 4096

    @property
    def covariances(self):
        """``(N, d, d)`` dense covariances, for code that wants the general shape.

        Refuses above :data:`DENSE_COVARIANCE_MAX_DIM`: the array would not fit in memory and
        nothing here needs it.  Use ``variances`` instead."""
        if self.dim > self.DENSE_COVARIANCE_MAX_DIM:
            raise MemoryError(
                f"a dense (N, d, d) covariance at d = {self.dim} would need "
                f"{self.n_components * self.dim ** 2 * 8 / 2**30:.0f} GiB. This family is "
                "isotropic -- use `variances` (component j is variances[j] * I).")
        return self.variances[:, None, None] * np.eye(self.dim)[None]

    def copy(self):
        return IsotropicMixture(self.means, self.variances)

    # ---------------------------------------------------------------- densities
    # The straightforward vectorisation of the two formulas below materialises an (M, N, d)
    # difference tensor.  That is fine in the paper's low-dimensional experiments and hopeless
    # at image scale: one conditional sample here is d = 3 x 256 x 256 = 196 608, where a single
    # (64, 8, d) temporary is 800 MB in float64.  Both methods therefore loop over the N
    # components -- N is small, and each step holds only an (M, d) temporary.  The arithmetic is
    # the SAME subtraction as the vectorised form, not an expanded |x|^2 - 2x.m + |m|^2 identity,
    # which would lose precision exactly where it matters (|x - m| much smaller than |x|).
    def log_component_densities(self, x):
        """``log[(1/N) N(x; m_j, eps_j I)]`` for ``x`` ``(M, d)`` -> ``(M, N)``."""
        x = np.asarray(x, dtype=float)
        quad = np.empty((x.shape[0], self.n_components), dtype=float)
        for j in range(self.n_components):
            diff = x - self.means[j][None, :]                            # (M, d)
            quad[:, j] = np.einsum("md,md->m", diff, diff) / self.variances[j]
        log_norm = -0.5 * self.dim * (_LOG2PI + np.log(self.variances))  # (N,)
        return log_norm[None, :] - 0.5 * quad - np.log(self.n_components)

    def log_prob(self, x):
        """``log q(x)`` for ``x`` ``(M, d)`` -> ``(M,)``."""
        from scipy.special import logsumexp

        return logsumexp(self.log_component_densities(x), axis=1)

    def score(self, x):
        """``grad_x log q(x)`` for ``x`` ``(M, d)`` -> ``(M, d)``.

        The responsibility-weighted sum of the per-component scores, computed through a
        softmax of the log-densities so that it is stable when a point is far from every
        component (where the naive ratio-of-sums underflows to 0/0)."""
        from scipy.special import softmax

        x = np.asarray(x, dtype=float)
        r = softmax(self.log_component_densities(x), axis=1)             # (M, N)
        out = np.zeros_like(x)
        for j in range(self.n_components):
            out -= (r[:, j] / self.variances[j])[:, None] * (x - self.means[j][None, :])
        return out

    # ---------------------------------------------------------------- sampling
    def sample_per_component(self, noise):
        """``m_j + sqrt(eps_j) z`` for shared standard noise ``(B, d)`` -> ``(N, B, d)``.

        One noise block is shared across components, as in the reference implementation: the
        same ``z`` reaches every component, which couples their Monte-Carlo errors and reduces
        the variance of *differences* between components."""
        noise = np.asarray(noise, dtype=float)
        if noise.ndim != 2 or noise.shape[1] != self.dim:
            raise ValueError(f"noise must be (B, {self.dim}), got {noise.shape}")
        return self.means[:, None, :] + np.sqrt(self.variances)[:, None, None] * noise[None, :, :]

    def sample(self, n, rng):
        """``n`` ancestral draws from the mixture -> ``(n, d)``."""
        rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        j = rng.integers(0, self.n_components, size=n)
        return self.means[j] + np.sqrt(self.variances[j])[:, None] * rng.standard_normal((n, self.dim))

    def reverse_kl(self, target, n, rng):
        """Monte-Carlo ``KL(q | pi)`` up to the additive ``log Z`` of an unnormalised target."""
        x = self.sample(n, rng)
        return float(np.mean(self.log_prob(x) - np.asarray(target.log_prob(x)).reshape(-1)))


# ======================================================================================
# gradients -- paper Proposition 3.1
# ======================================================================================
def mixture_gradients(q, target, noise):
    """Monte-Carlo ``(grad_means, grad_variances)`` of ``F = KL(q | pi)``.  Paper Prop. 3.1.

    ``noise`` is a ``(B, d)`` standard-normal block shared across components (see
    :meth:`IsotropicMixture.sample_per_component`).  Returns ``(N, d)`` and ``(N,)``.

    Both expectations are taken under component ``j``'s own Gaussian, and both integrands
    involve ``grad log(q/pi) = score_q - score_pi`` -- which is what couples the components to
    one another: a component's gradient depends on where all the others are, through ``q``.
    """
    samples = q.sample_per_component(noise)                              # (N, B, d)
    N, B, d = samples.shape
    flat = samples.reshape(N * B, d)

    score_q = q.score(flat).reshape(N, B, d)                             # grad log q
    score_pi = np.asarray(target.score(flat), dtype=float).reshape(N, B, d)
    if not np.all(np.isfinite(score_pi)):
        raise FloatingPointError("the target score returned non-finite values")
    delta = score_q - score_pi                                           # grad log(q/pi)
    centered = samples - q.means[:, None, :]                             # x - m_j

    grad_means = delta.mean(axis=1) / N                                  # (N, d)
    grad_variances = np.einsum("nbd,nbd->nb", delta, centered).mean(axis=1) / (2.0 * N * q.variances)
    return grad_means, grad_variances


# ======================================================================================
# the three variance geometries
# ======================================================================================
def _scaled_variance_gradient(grad_variances, n_components, dim, step):
    """The quantity ``(2 N gamma / d) grad_eps F`` shared by Eq. (12), (13) and (14)."""
    return (2.0 * n_components * step / dim) * np.asarray(grad_variances, dtype=float)


def ibw_variance_update(variances, grad_variances, n_components, dim, step):
    """Eq. (12): ``eps <- (1 - (2 N gamma / d) grad_eps F)^2 eps``.  Positive by construction."""
    u = _scaled_variance_gradient(grad_variances, n_components, dim, step)
    return (1.0 - u) ** 2 * np.asarray(variances, dtype=float)


def md_variance_update(variances, grad_variances, n_components, dim, step):
    """Eq. (13), entropic mirror descent: ``eps <- eps exp(-(2 N gamma / d) grad_eps F)``."""
    u = _scaled_variance_gradient(grad_variances, n_components, dim, step)
    return np.asarray(variances, dtype=float) * np.exp(-u)


def ngd_update(means, variances, grad_means, grad_variances, n_components, dim, step):
    """Eq. (14), natural gradient descent.  Returns ``(means, variances, ok)``.

    ``1/eps <- 1/eps + (2 N gamma / d) grad_eps F`` has no positivity guarantee (paper Sec. 5),
    and the mean step is preconditioned by the NEW variance rather than being the plain (GD)
    step, so the two updates do not decouple.  ``ok`` is False when a variance would leave
    ``R_+``; the caller decides what to do, and :func:`fit` stops."""
    u = _scaled_variance_gradient(grad_variances, n_components, dim, step)
    inv_new = 1.0 / np.asarray(variances, dtype=float) + u
    ok = bool(np.all(inv_new > 0))
    if not ok:
        return means, variances, False
    new_variances = 1.0 / inv_new
    new_means = means - step * n_components * new_variances[:, None] * grad_means
    return new_means, new_variances, True


# ======================================================================================
# Algorithm 1
# ======================================================================================
def fit(target, n_components=5, dim=None, method="ibw", n_iterations=1000, step_size=0.1,
        n_grad_samples=10, init_means=None, init_variances=None, init_spread=10.0,
        init_variance=1.0, seed=0, common_noise=False, kl_every=None, n_kl_samples=1000,
        callback=None, store_trace=None, fixed_noise=None):
    """Algorithm 1: fit a uniform-weight isotropic Gaussian mixture to ``target`` by reverse KL.

    ``target`` needs ``score(x)`` mapping ``(M, d) -> (M, d)``; ``log_prob(x) -> (M,)`` is used
    only for the optional KL diagnostic, and may be unnormalised.

    Parameters that are modelling choices rather than plumbing:

    ``method``         ``"ibw"`` (Eq. 12), ``"md"`` (Eq. 13) or ``"ngd"`` (Eq. 14);
    ``step_size``      the paper's ``gamma`` -- note the updates carry their own factors of
                       ``N`` and ``d``, so ``gamma`` is NOT a bare learning rate;
    ``n_grad_samples`` ``B``, the Monte-Carlo budget per component per iteration (the paper's
                       experiments use ``B_grad = 10``, Sec. E.2);
    ``init_spread`` /  the paper's ``s`` and ``r`` (Sec. E.2): means drawn uniformly in
    ``init_variance``  ``[-s, s]^d``, every variance set to ``r``;
    ``common_noise``   reuse ONE noise block for every iteration (common random numbers) instead
                       of drawing a fresh one -- both are available in the reference code;
    ``fixed_noise``    an explicit ``(B, d)`` block to reuse every iteration.  Supplying the same
                       block to two implementations makes their trajectories comparable exactly,
                       which is how ``tests/test_ibw_gpu_parity.py`` pins the torch backend.

    ``store_trace`` keeps the per-iteration parameter snapshots.  It defaults to True only when
    they fit comfortably in memory: the mean trace is ``n_iterations x N x d``, which is 3.8 TB
    for 300 iterations of 8 components at image scale, so it switches itself off above
    :data:`TRACE_MAX_ELEMENTS` and ``means_trace`` comes back None.  ``variances_trace`` is
    ``n_iterations x N`` and is always kept.

    Returns a dict with the fitted ``means`` / ``variances`` / ``weights``, the per-iteration
    ``means_trace`` / ``variances_trace`` (pre-update snapshots, so index 0 is the init), the
    reverse-KL diagnostic ``kls`` at ``kl_iters``, and ``stopped_early`` / ``n_iterations_run``.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    rng = np.random.default_rng(seed)

    # ---- initialisation (paper Sec. E.2) -------------------------------------------------
    if init_means is None:
        if dim is None:
            dim = getattr(target, "dim", None)
        if dim is None:
            raise ValueError("pass dim= or init_means= (the target carries no dimension)")
        init_means = rng.uniform(-init_spread, init_spread, size=(n_components, dim))
    init_means = np.asarray(init_means, dtype=float)
    n_components, dim = init_means.shape
    if init_variances is None:
        init_variances = np.full(n_components, float(init_variance))
    q = IsotropicMixture(init_means, init_variances)

    if fixed_noise is not None:
        fixed_noise = np.asarray(fixed_noise, dtype=float)
        if fixed_noise.shape != (n_grad_samples, dim):
            raise ValueError(f"fixed_noise must be ({n_grad_samples}, {dim}), got {fixed_noise.shape}")
        common_noise = True
    elif common_noise:
        fixed_noise = rng.standard_normal((n_grad_samples, dim))
    kl_every = max(1, n_iterations // 10) if kl_every is None else kl_every
    kl_rng_seed = int(rng.integers(0, 2**31 - 1))

    if store_trace is None:
        store_trace = n_iterations * n_components * dim <= TRACE_MAX_ELEMENTS
    means_trace = np.empty((n_iterations, n_components, dim)) if store_trace else None
    var_trace = np.empty((n_iterations, n_components))
    kls, kl_iters = [], []
    stopped_early = None

    n_run = 0
    for it in range(n_iterations):
        if store_trace:
            means_trace[it] = q.means                  # pre-update snapshot
        var_trace[it] = q.variances
        n_run = it + 1
        if kl_every and (it % kl_every == 0 or it == n_iterations - 1):
            kl_iters.append(it)
            # the SAME stream at every checkpoint, so the trace compares fits, not MC luck
            kls.append(q.reverse_kl(target, n_kl_samples, np.random.default_rng(kl_rng_seed)))

        noise = fixed_noise if common_noise else rng.standard_normal((n_grad_samples, dim))
        grad_means, grad_variances = mixture_gradients(q, target, noise)

        if method == "ngd":
            new_means, new_variances, ok = ngd_update(
                q.means, q.variances, grad_means, grad_variances, n_components, dim, step_size)
            if not ok:
                stopped_early = (f"NGD produced a non-positive variance at iteration {it}: "
                                 "Eq. (14) has no positivity guarantee (paper Sec. 5). "
                                 "Lower step_size, or use method='ibw' / 'md', which do.")
                means_trace = means_trace[:it + 1] if store_trace else None
                var_trace = var_trace[:it + 1]
                break
        else:
            new_means = q.means - step_size * n_components * grad_means          # (GD)
            update = ibw_variance_update if method == "ibw" else md_variance_update
            new_variances = update(q.variances, grad_variances, n_components, dim, step_size)
            if not np.all(new_variances > 0) or not np.all(np.isfinite(new_means)):
                stopped_early = (f"non-finite or non-positive iterate at iteration {it} "
                                 f"(method={method}, step_size={step_size})")
                means_trace = means_trace[:it + 1] if store_trace else None
                var_trace = var_trace[:it + 1]
                break

        q = IsotropicMixture(new_means, new_variances)
        if callback is not None:
            callback(it, q)

    return dict(
        means=q.means, variances=q.variances, weights=q.weights,
        means_trace=means_trace, variances_trace=var_trace,
        kls=np.asarray(kls), kl_iters=np.asarray(kl_iters, dtype=int),
        n_iterations_run=len(var_trace), stopped_early=stopped_early,
        settings=dict(method=method, step_size=step_size, n_grad_samples=n_grad_samples,
                      n_components=n_components, dim=dim, seed=seed, common_noise=common_noise),
    )
