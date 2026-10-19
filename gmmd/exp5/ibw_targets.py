"""Stage A, offline: IBW fits of the per-anchor conditional of each interval, K = 8.

Target.  For interval ``j`` with start state ``x`` (time ``t``) and end state at grid time ``s``
(``s = eps`` for the terminal interval), IBW descends ``KL(q || p)`` for

    p(y | x) ∝ p_s(y) N(x; y, (sigma_t^2 - sigma_s^2) I),
    grad_y log p(y | x) = s_theta(y, s) + (x - y) / (sigma_t^2 - sigma_s^2),

the existing VE Bayes construction (:mod:`gmmd.conditional_target`, batched in
:class:`gmmd.ibw_batched.VEConditionalScores`).  It is exact for the continuous-time reversal
under the learned marginal score, and it is NOT the density of the discretised 200-step teacher
kernel; stage-A evaluation measures that gap against teacher rollouts.

Family and update.  Unchanged :func:`gmmd.ibw_batched.fit_batched`: uniform weights 1/K (not
learned), one isotropic variance per component, means by (GD), variances by IBW Eq. (12).

Step size.  The interval's noise scale spans ~10 orders of magnitude over the trajectory, and
the updates are not scale-free: ``m <- m - gamma E[grad log(q/p)]``.  For a Gaussian target the
precision is at most ``1/v_anc`` with ``v_anc = sigma_s^2 gap / sigma_t^2`` (because
``-grad^2 log p_s <= I / sigma_s^2``), so ``gamma = c v_anc`` makes the stiffest direction
converge at the same rate ``c`` in every interval and is stable for ``c < 2``.  Only the scalar
``gamma`` changes per interval; Eq. (12) and (GD) are untouched.

Initialisations (all teacher-SAMPLE-free; the score network is the only teacher information,
as in the IBW updates themselves):

* ``prior_free``: the repository default, ``m_k = x + spread sqrt(gap) z_k``, ``v_k = gap``;
* ``principal``: components ``mu ± r sqrt(lambda_i) e_i`` along the top ``K/2`` eigenvectors of
  the Tweedie posterior covariance ``Cov[y | x] = gap (I + gap J)``, ``J = d s_theta(x, t)/dx``
  (second-order Tweedie for the VE forward kernel), around the posterior mean
  ``mu = x + gap s_theta(x, t)``.  These are the directions of largest conditional
  uncertainty, so a far spread along them places components in different regions of the
  conditional's density rather than in the off-manifold directions a random spread uses;
* ``random_far``: ``m_k = mu + r sqrt(d v0) u_k`` with random unit ``u_k`` (the literal "far
  apart" spread, kept as the control for ``principal``).
"""

from __future__ import annotations

import contextlib
import time

import numpy as np
import torch

from gmmd import ibw_batched
from gmmd.exp5.grid import seed_of

__all__ = ["INIT_SCHEMES", "tf32", "make_target", "tweedie_mean", "posterior_principal_directions",
           "initial_mixture", "fit_interval", "component_stats", "isotropic_log_prob",
           "sample_mixture"]

INIT_SCHEMES = ("prior_free", "principal", "random_far")


@contextlib.contextmanager
def tf32(enabled):
    """Set cuDNN/matmul TF32 for the block, then restore -- precision is a per-stage choice."""
    old = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


def make_target(teacher, iv, x_start, batch_size=64):
    """The batched VE Bayes conditional of interval ``iv`` at start states ``x_start`` (A,C,H,W)."""
    return ibw_batched.VEConditionalScores(teacher, x_start.to(teacher.device), iv.t_start,
                                           iv.t_end_state, batch_size=batch_size)


def _scores(teacher, x, t, batch_size):
    out = torch.empty_like(x)
    vec = torch.full((x.shape[0],), float(t), device=x.device)
    with torch.no_grad():
        for j in range(0, x.shape[0], batch_size):
            out[j:j + batch_size] = teacher._score_fn(x[j:j + batch_size], vec[j:j + batch_size])
    return out


def tweedie_mean(teacher, iv, x_start, batch_size=64):
    """``E[y | x] = x + gap s_theta(x, t)`` (first-order Tweedie for the forward kernel). (A, d)."""
    x = x_start.to(teacher.device, torch.float32)
    s = _scores(teacher, x, iv.t_start, batch_size)
    return (x + iv.gap * s).reshape(x.shape[0], -1), x.shape[0]


def posterior_principal_directions(teacher, iv, x_start, q, n_iter=8, oversample=2,
                                   rel_h=0.05, seeds=None, batch_size=64):
    """Top-``q`` eigenpairs of ``Cov[y | x] = gap (I + gap J)`` at each start state.

    Subspace iteration with ``q + oversample`` vectors and central differences
    ``J v ~ (s(x + h v) - s(x - h v)) / (2 h)``, ``h = rel_h sigma_t``, then Rayleigh-Ritz on the
    symmetric part.  MUST run in full fp32: the difference is ~1e-4 of the score per coordinate,
    below TF32's rounding.  Returns ``(eigvals (A, q), eigvecs (A, q, d), nfe)``, eigenvalues of
    the covariance itself (variance along each direction), descending."""
    if torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("principal directions need fp32 scores (wrap in tf32(False))")
    x = x_start.to(teacher.device, torch.float32)
    A = x.shape[0]
    shape = tuple(x.shape[1:])
    d = int(np.prod(shape))
    p = int(q) + int(oversample)
    h = float(rel_h) * iv.sigma_start
    seeds = [seed_of(61, a) for a in range(A)] if seeds is None else seeds
    V = torch.stack([torch.randn((d, p), generator=torch.Generator(device=x.device).manual_seed(int(s)),
                                 device=x.device) for s in seeds])                   # (A, d, p)
    V, _ = torch.linalg.qr(V)
    nfe = 0

    def apply_cov(V):
        nonlocal nfe
        pert = h * V.transpose(1, 2).reshape(A, p, *shape)                           # (A, p, ...)
        xs = torch.cat([x[:, None] + pert, x[:, None] - pert], 1).reshape(A * 2 * p, *shape)
        s = _scores(teacher, xs, iv.t_start, batch_size).reshape(A, 2, p, d)
        nfe += A * 2 * p
        JV = ((s[:, 0] - s[:, 1]) / (2 * h)).transpose(1, 2)                          # (A, d, p)
        return iv.gap * (V + iv.gap * JV)

    for _ in range(int(n_iter)):
        V, _ = torch.linalg.qr(apply_cov(V))
    W = apply_cov(V)
    H = V.transpose(1, 2) @ W
    H = 0.5 * (H + H.transpose(1, 2))
    vals, vecs = torch.linalg.eigh(H.double())
    vals, vecs = vals.flip(-1), vecs.flip(-1)                                         # descending
    E = (V.double() @ vecs).float()                                                   # (A, d, p)
    return vals[:, :q].float(), E[:, :, :q].transpose(1, 2).contiguous(), nfe


def initial_mixture(scheme, teacher, iv, x_start, K, seeds, *, spread=1.0, var="gap",
                    batch_size=64, principal=None):
    """``(means (A, K, d), variances (A, K), info)`` on the teacher device.

    ``var``: the initial variance, ``"gap"``, ``"v_anc"`` or a float multiple of ``v_anc``.
    ``principal``: precomputed ``(eigvals, eigvecs)`` for the ``principal`` scheme (fp32)."""
    if scheme not in INIT_SCHEMES:
        raise ValueError(f"init scheme must be one of {INIT_SCHEMES}, got {scheme!r}")
    dev = teacher.device
    x = x_start.to(dev, torch.float32)
    A = x.shape[0]
    d = int(np.prod(x.shape[1:]))
    v0 = iv.gap if var == "gap" else iv.v_anc if var == "v_anc" else float(var) * iv.v_anc
    gens = [torch.Generator(device=dev).manual_seed(int(s)) for s in seeds]
    info = dict(scheme=scheme, spread=float(spread), var=var, v0=float(v0), nfe=0)
    if scheme == "prior_free":
        z = torch.stack([torch.randn((K, d), generator=g, device=dev) for g in gens])
        means = x.reshape(A, 1, d) + float(spread) * float(np.sqrt(iv.gap)) * z
    else:
        mu, n = tweedie_mean(teacher, iv, x, batch_size)
        info["nfe"] += n
        if scheme == "principal":
            if K % 2:
                raise ValueError("the principal scheme places components in ± pairs; K must be even")
            if principal is None:
                raise ValueError("pass principal=(eigvals, eigvecs) computed in fp32")
            lam, E = principal
            lam, E = lam.to(dev)[:, :K // 2], E.to(dev)[:, :K // 2]
            off = float(spread) * lam.clamp(min=0).sqrt()[..., None] * E                 # (A, K/2, d)
            means = torch.stack([mu[:, None] + off, mu[:, None] - off], 2).reshape(A, K, d)
            info["eigvals"] = lam.double().cpu().numpy()
        else:                                                                           # random_far
            u = torch.stack([torch.randn((K, d), generator=g, device=dev) for g in gens])
            u = u / u.norm(dim=-1, keepdim=True)
            means = mu[:, None] + float(spread) * float(np.sqrt(d * v0)) * u
    variances = torch.full((A, K), float(v0), device=dev)
    return means, variances, info


def fit_interval(teacher, iv, x_start, init_means, init_variances, noise_seeds, *, step_c,
                 n_iterations, n_grad_samples=4, snapshot_at=(), diag_every=25, batch_size=64):
    """Batched IBW on interval ``iv`` for ``A`` anchors; ``gamma = step_c * v_anc``."""
    target = make_target(teacher, iv, x_start, batch_size)
    res = ibw_batched.fit_batched(target, init_means, init_variances, noise_seeds,
                                  n_iterations=int(n_iterations), step_size=float(step_c) * iv.v_anc,
                                  n_grad_samples=int(n_grad_samples), snapshot_at=snapshot_at,
                                  device=teacher.device, diag_every=diag_every)
    res["nfe"] = int(target.nfe)
    res["target"] = target.describe()
    return res


# ======================================================================================
# mixture utilities
# ======================================================================================
def component_stats(means, variances, sep_threshold=1.0):
    """Separation and effective-component diagnostics of one isotropic mixture.

    ``sep_ij = ||m_i - m_j|| / (sqrt(v_i) + sqrt(v_j))``.  Components closer than
    ``sep_threshold`` are merged (single linkage); ``n_effective`` counts the groups.  At
    d = 196 608, separations are relative diagnostics only (README of exp3)."""
    m = torch.as_tensor(means).double()
    v = torch.as_tensor(variances).double().reshape(-1)
    K = m.shape[0]
    D = torch.cdist(m[None], m[None])[0]
    sd = v.sqrt()
    sep = D / (sd[:, None] + sd[None, :])
    parent = list(range(K))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(K):
        for j in range(i + 1, K):
            if sep[i, j] < sep_threshold:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(K):
        groups.setdefault(find(i), []).append(i)
    sizes = sorted((len(g) for g in groups.values()), reverse=True)
    w = np.asarray(sizes, float) / K
    iu = torch.triu_indices(K, K, 1)
    off = sep[iu[0], iu[1]] if K > 1 else torch.zeros(1, dtype=torch.float64)
    return dict(n_effective=len(groups), group_sizes=sizes,
                weight_perplexity=float(np.exp(-(w * np.log(w)).sum())),
                sep_min=float(off.min()), sep_median=float(off.median()), sep_max=float(off.max()),
                dist_min=float(D[iu[0], iu[1]].min()) if K > 1 else 0.0,
                var_min=float(v.min()), var_max=float(v.max()), var_mean=float(v.mean()))


def isotropic_log_prob(z, means, variances, log_weights=None):
    """``log q(z)`` of an isotropic mixture: per-component log densities SUMMED over the d
    coordinates, then log-sum-exp over components.  ``z`` (M, d), ``means`` (K, d) -> (M,)."""
    z = z.double()
    m = torch.as_tensor(means, device=z.device).double()
    v = torch.as_tensor(variances, device=z.device).double().reshape(-1)
    K, d = m.shape
    lw = (torch.full((K,), -np.log(K), dtype=torch.float64, device=z.device) if log_weights is None
          else torch.as_tensor(log_weights, device=z.device).double())
    comp = torch.stack([((z - m[k]) ** 2).sum(-1) for k in range(K)], -1)              # (M, K)
    logn = -0.5 * (d * (np.log(2 * np.pi) + torch.log(v))[None] + comp / v[None])
    return torch.logsumexp(lw[None] + logn, -1)


def sample_mixture(means, variances, n, generator, weights=None):
    """``n`` ancestral draws of one isotropic mixture: a component, then fresh Gaussian noise."""
    m = torch.as_tensor(means)
    v = torch.as_tensor(variances, device=m.device).reshape(-1)
    K, d = m.shape
    p = (torch.full((K,), 1.0 / K, device=m.device) if weights is None
         else torch.as_tensor(weights, device=m.device, dtype=torch.float32))
    k = torch.multinomial(p, n, replacement=True, generator=generator)
    z = torch.randn((n, d), device=m.device, dtype=m.dtype, generator=generator)
    return m[k] + v[k].sqrt()[:, None].to(m.dtype) * z, k


def timed(device):
    """Seconds of a synchronised block (helper for per-stage cost records)."""
    class _T:
        def __enter__(self):
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            self.seconds = time.perf_counter() - self.t0
            return False
    return _T()
