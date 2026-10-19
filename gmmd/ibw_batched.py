"""IBW for many independent targets at once -- one mixture per anchor, fitted in lockstep.

Replacing a diffusion interval needs one fit per generated image: every new ``x_t`` defines a
new conditional ``p(x_s | x_t)``.  Fitting them one at a time pushes ``K x B`` images (4 at the
paper's K = 1, B = 4) through the score network per iteration, which leaves most of the GPU
idle.  Here ``A`` anchors run the same iteration together, so one network call carries
``A x K x B`` images.  Nothing couples the anchors: each has its own mixture, its own noise
generator and its own target, and the update is :func:`gmmd.ibw_gpu.fit`'s Eq. (12) applied
per anchor.  With the same seed, anchor ``a`` reproduces ``ibw_gpu.fit`` on that target
(``tests/test_interval_replacement.py``).

An anchor whose iterate turns non-finite or non-positive is frozen at its last valid state and
flagged in ``stopped_early``; the others continue.
"""

from __future__ import annotations

import time

import numpy as np
import torch

__all__ = ["fit_batched", "sample_batched", "VEConditionalScores", "ITERATION_COUNTER"]

# Total IBW iterations run in this process (all anchors of one iteration count once).  The exp5
# standalone sampler reads it before and after generation to certify that NO online fitting ran.
ITERATION_COUNTER = [0]


class VEConditionalScores:
    """Conditional scores ``s_theta(x, s) + (x_t[a] - x) / (sigma_t^2 - sigma_s^2)`` for A anchors.

    The batched form of :class:`gmmd.conditional_target.VEConditionalTarget` -- same target, same
    arithmetic -- with the network called in chunks of ``batch_size`` images.
    """

    def __init__(self, teacher, x_t, t, s, batch_size=64):
        from gmmd.conditional_target import VEConditionalTarget

        ref = VEConditionalTarget(teacher, x_t[0].detach().cpu().numpy(), t, s)
        self.t, self.s, self.gap = ref.t, ref.s, ref.gap
        self.sigma_t, self.sigma_s = ref.sigma_t, ref.sigma_s
        self.teacher = teacher
        self.image_shape = tuple(teacher.image_shape)
        self.dim = int(np.prod(self.image_shape))
        self.x_t = x_t.reshape(x_t.shape[0], -1).to(torch.float32)
        self.batch_size = int(batch_size)
        self.nfe = 0
        self._vec = None

    def __call__(self, x, anchor):
        """``x`` (M, d) on device, ``anchor`` (M,) long -> (M, d)."""
        out = torch.empty_like(x)
        img = x.reshape(-1, *self.image_shape)
        for j in range(0, x.shape[0], self.batch_size):
            xb = img[j:j + self.batch_size].to(torch.float32)
            vec = torch.full((xb.shape[0],), self.s, device=x.device)
            with torch.no_grad():
                sb = self.teacher._score_fn(xb, vec)
            self.nfe += int(xb.shape[0])
            out[j:j + self.batch_size] = sb.reshape(xb.shape[0], -1).to(x.dtype)
        return out + (self.x_t[anchor].to(x.dtype) - x) / self.gap

    def describe(self):
        return dict(kind="ve_bayes_conditional_batched", t=self.t, s=self.s,
                    sigma_t=self.sigma_t, sigma_s=self.sigma_s, gap=self.gap, dim=self.dim)


def _mixture_score(means, variances, x):
    """Score of each anchor's uniform isotropic mixture. means (A,K,d), variances (A,K), x (A,M,d)."""
    A, K, d = means.shape
    quad = torch.stack([((x - means[:, j, None, :]) ** 2).sum(-1) / variances[:, j, None]
                        for j in range(K)], dim=-1)                                   # (A,M,K)
    log_norm = -0.5 * d * (float(np.log(2 * np.pi)) + torch.log(variances))          # (A,K)
    r = torch.softmax(log_norm[:, None, :] - 0.5 * quad, dim=-1)                      # (A,M,K)
    out = torch.zeros_like(x)
    for j in range(K):
        out -= (r[..., j] / variances[:, j, None])[..., None] * (x - means[:, j, None, :])
    return out


def fit_batched(score_fn, init_means, init_variances, seeds, n_iterations=300, step_size=0.1,
                n_grad_samples=4, snapshot_at=(), dtype=torch.float32, device="cuda",
                diag_every=0):
    """Run IBW (Eq. 12) on ``A`` independent targets.

    ``score_fn(x (M, d), anchor (M,)) -> (M, d)``; ``init_means`` (A, K, d), ``init_variances``
    (A, K); ``seeds`` one per anchor (the noise generator ``ibw_gpu.fit`` would build from it).
    ``snapshot_at``: iteration counts at which to keep a copy of every anchor's mixture.
    ``diag_every``: if > 0, every that many iterations record each anchor's variances and the
    displacement of its means since the previous record (convergence diagnostics; no effect on
    the iterates).

    Returns a dict with final ``means`` (A,K,d) / ``variances`` (A,K) as tensors on device,
    ``snapshots`` {iteration: (means, variances)}, ``grad_norms`` (n_iterations, A),
    ``stopped_early`` (list per anchor, None or a message), ``runtime_s``, and with
    ``diag_every`` a ``diagnostics`` dict: ``iterations`` (n_rec,), ``variances`` (n_rec, A, K),
    ``mean_drift`` (n_rec, A) = ``||m(it) - m(it - diag_every)||_F / sqrt(K d mean(v))``.
    """
    device = torch.device(device)
    means = torch.as_tensor(init_means, dtype=dtype, device=device).clone()
    variances = torch.as_tensor(init_variances, dtype=dtype, device=device).clone()
    A, K, d = means.shape
    B = int(n_grad_samples)
    gens = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    anchor_idx = torch.arange(A, device=device).repeat_interleave(K * B)
    active = torch.ones(A, dtype=torch.bool, device=device)
    stopped = [None] * A
    grad_norms = np.full((n_iterations, A), np.nan)
    snapshot_at = {int(v) for v in snapshot_at}
    snapshots = {}
    diag_every = int(diag_every)
    diag = dict(iterations=[0], variances=[variances.double().cpu().numpy()],
                mean_drift=[np.full(A, np.nan)]) if diag_every > 0 else None
    last_means = means.clone() if diag_every > 0 else None
    t0 = time.perf_counter()

    for it in range(n_iterations):
        ITERATION_COUNTER[0] += 1
        noise = torch.stack([torch.randn((B, d), device=device, dtype=dtype, generator=g)
                             for g in gens])                                           # (A,B,d)
        samples = means[:, :, None, :] + variances.sqrt()[:, :, None, None] * noise[:, None]
        flat = samples.reshape(A, K * B, d)                                            # (A,KB,d)
        score_q = _mixture_score(means, variances, flat).reshape(A, K, B, d)
        score_pi = score_fn(flat.reshape(A * K * B, d), anchor_idx).reshape(A, K, B, d)
        delta = score_q - score_pi
        centered = samples - means[:, :, None, :]
        grad_means = delta.mean(dim=2) / K                                             # (A,K,d)
        grad_vars = (delta * centered).sum(-1).mean(-1) / (2.0 * K * variances)       # (A,K)
        grad_norms[it] = grad_means.reshape(A, -1).norm(dim=1).double().cpu().numpy()

        u = (2.0 * K * step_size / d) * grad_vars
        new_vars = (1.0 - u) ** 2 * variances
        new_means = means - step_size * K * grad_means
        ok = (torch.isfinite(new_means).reshape(A, -1).all(1) & (new_vars > 0).all(1)
              & torch.isfinite(new_vars).all(1))
        bad = active & ~ok
        for a in torch.nonzero(bad).flatten().tolist():
            stopped[a] = f"non-finite or non-positive iterate at iteration {it}"
        active = active & ok
        sel = active[:, None]
        means = torch.where(sel[:, :, None], new_means, means)
        variances = torch.where(sel, new_vars, variances)
        if (it + 1) in snapshot_at:
            snapshots[it + 1] = (means.clone(), variances.clone())
        if diag_every > 0 and (it + 1) % diag_every == 0:
            scale = (K * d * variances.mean(1)).sqrt()
            drift = (means - last_means).reshape(A, -1).norm(dim=1) / scale
            diag["iterations"].append(it + 1)
            diag["variances"].append(variances.double().cpu().numpy())
            diag["mean_drift"].append(drift.double().cpu().numpy())
            last_means = means.clone()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    out = dict(means=means, variances=variances, snapshots=snapshots, grad_norms=grad_norms,
               stopped_early=stopped, runtime_s=time.perf_counter() - t0,
               settings=dict(method="ibw", step_size=step_size, n_grad_samples=B,
                             n_components=K, dim=d, n_iterations=n_iterations,
                             backend="torch_batched", dtype=str(dtype)))
    if diag is not None:
        out["diagnostics"] = {k: np.asarray(v) for k, v in diag.items()}
    return out


def sample_batched(means, variances, n, gens):
    """``n`` ancestral draws from each anchor's mixture: component uniform, then ``m + sqrt(v) z``.

    ``gens``: one CUDA generator per anchor.  Returns (A, n, d).  A random draw, never the mean.
    """
    A, K, d = means.shape
    out = torch.empty((A, n, d), device=means.device, dtype=means.dtype)
    for a, g in enumerate(gens):
        j = torch.randint(0, K, (n,), device=means.device, generator=g)
        z = torch.randn((n, d), device=means.device, dtype=means.dtype, generator=g)
        out[a] = means[a, j] + variances[a, j].sqrt()[:, None] * z
    return out
