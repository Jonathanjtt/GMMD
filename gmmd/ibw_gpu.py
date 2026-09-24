"""A torch/GPU backend for :mod:`gmmd.ibw`, for targets whose score is a neural network.

Same algorithm, same equations, different execution. :mod:`gmmd.ibw` is the *reference*: NumPy,
readable, and checked against the authors' own implementation in
``tests/test_ibw_reference_parity.py``. This module is the one to use when the target score is a
GPU network and the dimension is an image rather than a toy, and
``tests/test_ibw_gpu_parity.py`` pins the two to each other.

Why a second implementation rather than reusing an existing GPU one. The sibling project GMMVI
carries a JAX IBW that runs the entire optimisation inside a single ``jax.jit`` + ``lax.scan``
with no host round-trip, which is the right design -- but it calls the target's ``score_fn``
*inside* that scan, so the score must be traceable JAX. The teacher here is Song et al.'s NCSN++
in PyTorch, which cannot be traced into ``lax.scan``. Hence torch, where the network can simply
be called.

What this buys at image scale (d = 196 608): the mixture's own linear algebra runs on the GPU in
float32 next to the network instead of in float64 on the host, and the samples never leave the
device -- the NumPy path moves ``K x B x d`` floats across PCIe twice per iteration, which at
K = 8, B = 4 is 100 MB of traffic per step for arithmetic that takes microseconds on the GPU.

The mathematics is unchanged and is documented in :mod:`gmmd.ibw`. As there, both methods loop
over the ``N`` components rather than forming an ``(M, N, d)`` tensor, which at image scale would
be hundreds of gigabytes.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = ["TorchIsotropicMixture", "mixture_gradients", "fit"]

_LOG2PI = float(np.log(2.0 * np.pi))


class TorchIsotropicMixture:
    """``q = (1/N) sum_j N(m_j, eps_j I)`` held on a torch device. Mirrors ``ibw.IsotropicMixture``."""

    def __init__(self, means, variances, device=None, dtype=None):
        import torch

        dtype = torch.float32 if dtype is None else dtype
        self.means = torch.as_tensor(np.asarray(means), dtype=dtype, device=device).clone()
        self.variances = torch.as_tensor(np.asarray(variances), dtype=dtype,
                                         device=device).reshape(-1).clone()
        if self.means.ndim != 2:
            raise ValueError(f"means must be (N, d), got {tuple(self.means.shape)}")
        if self.variances.shape[0] != self.means.shape[0]:
            raise ValueError(f"{self.means.shape[0]} means but {self.variances.shape[0]} variances")
        if not bool((self.variances > 0).all()):
            raise ValueError("every variance must be strictly positive")
        self._torch = torch

    @property
    def n_components(self):
        return int(self.means.shape[0])

    @property
    def dim(self):
        return int(self.means.shape[1])

    @property
    def device(self):
        return self.means.device

    @property
    def weights(self):
        return self._torch.full((self.n_components,), 1.0 / self.n_components,
                                device=self.device, dtype=self.means.dtype)

    def log_component_densities(self, x):
        """``log[(1/N) N(x; m_j, eps_j I)]`` for ``x`` ``(M, d)`` -> ``(M, N)``."""
        torch = self._torch
        quad = torch.empty((x.shape[0], self.n_components), device=x.device, dtype=x.dtype)
        for j in range(self.n_components):
            diff = x - self.means[j][None, :]
            quad[:, j] = (diff * diff).sum(1) / self.variances[j]
        log_norm = -0.5 * self.dim * (_LOG2PI + torch.log(self.variances))
        return log_norm[None, :] - 0.5 * quad - float(np.log(self.n_components))

    def log_prob(self, x):
        return self._torch.logsumexp(self.log_component_densities(x), dim=1)

    def score(self, x):
        """``grad_x log q(x)`` for ``x`` ``(M, d)`` -> ``(M, d)``."""
        torch = self._torch
        r = torch.softmax(self.log_component_densities(x), dim=1)
        out = torch.zeros_like(x)
        for j in range(self.n_components):
            out -= (r[:, j] / self.variances[j])[:, None] * (x - self.means[j][None, :])
        return out

    def sample_per_component(self, noise):
        """``m_j + sqrt(eps_j) z`` for shared noise ``(B, d)`` -> ``(N, B, d)``."""
        return self.means[:, None, :] + self.variances.sqrt()[:, None, None] * noise[None, :, :]

    def sample(self, n, generator=None):
        """``n`` ancestral draws -> ``(n, d)``."""
        torch = self._torch
        j = torch.randint(0, self.n_components, (n,), device=self.device, generator=generator)
        z = torch.randn((n, self.dim), device=self.device, dtype=self.means.dtype,
                        generator=generator)
        return self.means[j] + self.variances[j].sqrt()[:, None] * z

    def to_numpy(self):
        return (self.means.double().cpu().numpy(), self.variances.double().cpu().numpy())


def mixture_gradients(q, target, noise):
    """Proposition 3.1 gradients, on device. ``target`` must expose ``score_torch``."""
    torch = q._torch
    samples = q.sample_per_component(noise)                       # (N, B, d)
    N, B, d = samples.shape
    flat = samples.reshape(N * B, d)

    score_q = q.score(flat).reshape(N, B, d)
    score_pi = target.score_torch(flat).reshape(N, B, d)
    if not bool(torch.isfinite(score_pi).all()):
        raise FloatingPointError("the target score returned non-finite values")
    delta = score_q - score_pi
    centered = samples - q.means[:, None, :]

    grad_means = delta.mean(dim=1) / N
    grad_variances = (delta * centered).sum(-1).mean(-1) / (2.0 * N * q.variances)
    return grad_means, grad_variances


def fit(target, n_components=5, method="ibw", n_iterations=300, step_size=0.1, n_grad_samples=4,
        init_means=None, init_variances=None, seed=0, device=None, dtype=None, log=None,
        log_every=50, fixed_noise=None):
    """IBW / MD Algorithm 1 on a torch device. Returns the same shape of dict as ``ibw.fit``.

    ``target`` needs ``score_torch(x)`` mapping ``(M, d) -> (M, d)`` torch tensors, and ``dim``.
    Only ``"ibw"`` (Eq. 12) and ``"md"`` (Eq. 13) are implemented here; NGD's positivity handling
    lives in the reference path, where it is cheap to inspect.

    ``fixed_noise`` is an explicit ``(B, d)`` block reused every iteration; handing the same one
    to this function and to :func:`gmmd.ibw.fit` makes the two trajectories comparable exactly,
    which is what ``tests/test_ibw_gpu_parity.py`` does.

    No reverse-KL trace: the targets this backend exists for have no tractable log-density.
    ``variances_trace`` and a gradient-norm trace are kept; the ``(n_iterations, N, d)`` mean
    trace is not, because it is terabytes at image scale.
    """
    import torch

    if method not in ("ibw", "md"):
        raise ValueError(f"the GPU backend implements 'ibw' and 'md'; got {method!r}. Use "
                         "gmmd.ibw.fit for NGD.")
    dtype = torch.float32 if dtype is None else dtype
    device = torch.device(device if device is not None else
                          (getattr(target, "device", None) or "cpu"))
    gen = torch.Generator(device=device).manual_seed(int(seed))
    if fixed_noise is not None:
        fixed_noise = torch.as_tensor(np.asarray(fixed_noise), dtype=dtype, device=device)

    q = TorchIsotropicMixture(init_means, init_variances, device=device, dtype=dtype)
    N, d = q.n_components, q.dim

    var_trace = np.empty((n_iterations, N))
    grad_norms = np.empty(n_iterations)
    stopped_early, n_run = None, 0
    t0 = time.perf_counter()

    for it in range(n_iterations):
        var_trace[it] = q.variances.double().cpu().numpy()
        n_run = it + 1
        noise = (fixed_noise if fixed_noise is not None else
                 torch.randn((n_grad_samples, d), device=device, dtype=dtype, generator=gen))
        grad_means, grad_variances = mixture_gradients(q, target, noise)
        grad_norms[it] = float(grad_means.norm())

        u = (2.0 * N * step_size / d) * grad_variances            # the shared scaled gradient
        if method == "ibw":
            new_variances = (1.0 - u) ** 2 * q.variances          # Eq. (12)
        else:
            new_variances = q.variances * torch.exp(-u)           # Eq. (13)
        new_means = q.means - step_size * N * grad_means          # (GD)

        if not bool(torch.isfinite(new_means).all() and (new_variances > 0).all()):
            stopped_early = (f"non-finite or non-positive iterate at iteration {it} "
                             f"(method={method}, step_size={step_size})")
            var_trace, grad_norms = var_trace[:it + 1], grad_norms[:it + 1]
            break
        q.means, q.variances = new_means, new_variances
        if log is not None and log_every and (it % log_every == 0 or it == n_iterations - 1):
            log(f"    it {it:4d}  |grad_m| {grad_norms[it]:10.3f}  "
                f"eps {np.round(var_trace[it], 3).tolist()}")

    means, variances = q.to_numpy()
    return dict(means=means, variances=variances,
                weights=np.full(N, 1.0 / N), variances_trace=var_trace, means_trace=None,
                grad_norms=grad_norms, kls=np.asarray([]), kl_iters=np.asarray([], dtype=int),
                n_iterations_run=n_run, stopped_early=stopped_early,
                runtime_s=time.perf_counter() - t0,
                settings=dict(method=method, step_size=step_size, n_grad_samples=n_grad_samples,
                              n_components=N, dim=d, seed=seed, backend="torch",
                              device=str(device), dtype=str(dtype)))
