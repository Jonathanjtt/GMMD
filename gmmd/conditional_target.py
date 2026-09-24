"""The conditional ``p_θ(x_s | x_t)`` as a target IBW can optimise -- and exactly which one it is.

Reading this before using it is not optional: there are two different objects that both deserve
the name ``p_θ(x_s | x_t)``, and this module implements the one the *sampler approximates*, not
the sampler's own kernel.

**(a) The model's conditional.**  For the VE SDE the forward process has no drift, so
``x_t | x_s ~ N(x_s, (σ_t² - σ_s²) I)`` exactly.  Bayes' rule then gives a closed form for the
score of the conditional:

    ∇_{x_s} log p(x_s | x_t) = ∇ log p_s(x_s) + (x_t - x_s) / (σ_t² - σ_s²)
                             ≈ s_θ(x_s, s)    + (x_t - x_s) / (σ_t² - σ_s²)

with ``s_θ`` the score network.  Every term is computable.  This is the conditional of the
continuous-time time reversal -- the object the reverse-SDE sampler is a discretisation of.

**(b) The sampler's transition kernel.**  What the H1 experiment actually drew from: a
composition of 40 (or 299) Gaussian predictor steps.  Its density and score have no closed form
beyond a single step, and adding the Langevin corrector changes it again.

This module implements **(a)**, because it is the one with a score, and IBW needs a score.  The
gap between them is discretisation error, and it is not assumed away: it is *measured*, by
comparing samples from the fitted ``q`` against the teacher samples drawn from (b), which the
H1 run already saved.  A short transition has fewer steps to accumulate that error, which is
one reason the short transition is the sensible place to start.

Three further caveats, none of them hidden:

* ``s_θ`` is a *learned* score.  A learned vector field need not be the exact gradient of any
  density, so the "conditional" above is exact only to the extent the network is.
* Only the SCORE is available, never the log-density: ``log p_s(x_s)`` is unknown.  So the
  reverse-KL objective value cannot be evaluated, only descended.  ``log_prob`` raises, and
  model selection has to be done on samples rather than on a KL number.
* ``σ_t² - σ_s²`` appears in a denominator, so the correction term blows up as ``s → t``.  That
  is real, not numerical: as the transition shrinks the conditional collapses onto ``x_t``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["VEConditionalTarget"]


class VEConditionalTarget:
    """``p_θ(x_s | x_t)`` for a VE SDE, exposing the closed-form conditional score of case (a).

    Consumed by :func:`gmmd.ibw.fit` through ``dim`` and ``score``; ``log_prob`` raises, so the
    caller must pass ``kl_every=0``.

    Parameters
    ----------
    teacher : gmmd.teachers.score_sde.teacher.ScoreSDETeacher
    x_t : array, the frozen conditioning tensor, shape ``teacher.image_shape``
    t, s : float, the diffusion times; ``s < t``, both snapped to the teacher's grid
    batch_size : int, how many points to push through the score network at once
    """

    def __init__(self, teacher, x_t, t, s, batch_size=16):
        import torch

        if not isinstance(teacher.sde, type(teacher.sde)) or type(teacher.sde).__name__ != "VESDE":
            raise NotImplementedError(
                f"the closed-form conditional score above is derived for the VE SDE (no drift, "
                f"x_t | x_s ~ N(x_s, (sigma_t^2 - sigma_s^2) I)); this teacher runs "
                f"{type(teacher.sde).__name__}, whose forward kernel has a mean coefficient and "
                "therefore a different Bayes correction. Derive it before using this class.")
        self.teacher = teacher
        self.batch_size = int(batch_size)
        self._torch = torch

        _, _, self.t, self.s = teacher.transition_indices(t, s)
        self.sigma_t = teacher.sigma(self.t)
        self.sigma_s = teacher.sigma(self.s)
        self.gap = self.sigma_t ** 2 - self.sigma_s ** 2          # variance of x_t given x_s
        if self.gap <= 0:
            raise ValueError(f"sigma_t^2 - sigma_s^2 = {self.gap} must be positive")

        x_t = np.asarray(x_t, dtype=np.float32).reshape(teacher.image_shape)
        self.image_shape = tuple(teacher.image_shape)
        self.dim = int(np.prod(self.image_shape))
        self.x_t = x_t.reshape(-1).astype(np.float64)             # flat, the IBW convention
        self._x_t_t = torch.as_tensor(x_t, device=teacher.device)[None]
        self.n_score_calls = 0

    # ------------------------------------------------------------------ the two pieces
    def marginal_score(self, x):
        """``s_θ(x, s)``: the score network alone, ``(M, d) -> (M, d)``."""
        torch = self._torch
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        for i in range(0, len(x), self.batch_size):
            chunk = x[i:i + self.batch_size]
            xb = torch.as_tensor(chunk.reshape(-1, *self.image_shape), dtype=torch.float32,
                                 device=self.teacher.device)
            with torch.no_grad():
                sb = self.teacher.score(xb, self.s)
            self.n_score_calls += len(chunk)
            out[i:i + self.batch_size] = sb.reshape(len(chunk), -1).double().cpu().numpy()
        return out

    def likelihood_score(self, x):
        """``(x_t - x_s) / (σ_t² - σ_s²)``: the forward-kernel term, no network involved."""
        return (self.x_t[None, :] - np.asarray(x, dtype=np.float64)) / self.gap

    def score(self, x):
        """``∇_{x_s} log p(x_s | x_t)`` -- the sum of the two terms above."""
        return self.marginal_score(x) + self.likelihood_score(x)

    # ------------------------------------------------------------------ torch / GPU path
    @property
    def device(self):
        return self.teacher.device

    def score_torch(self, x):
        """The same conditional score for ``(M, d)`` torch tensors, staying on the device.

        This is what :mod:`gmmd.ibw_gpu` calls. Identical arithmetic to :meth:`score`, but the
        samples never cross to the host: at d = 196 608 the NumPy path moves 100 MB per
        iteration each way for arithmetic that costs microseconds on the GPU.
        """
        torch = self._torch
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected (M, {self.dim}) tensor, got {tuple(x.shape)}")
        out = torch.empty_like(x)
        img = x.reshape(-1, *self.image_shape)
        for i in range(0, x.shape[0], self.batch_size):
            with torch.no_grad():
                sb = self.teacher.score(img[i:i + self.batch_size].to(torch.float32), self.s)
            self.n_score_calls += sb.shape[0]
            out[i:i + self.batch_size] = sb.reshape(sb.shape[0], -1).to(x.dtype)
        return out + (self._x_t_t.reshape(1, -1).to(x.dtype) - x) / self.gap

    def log_prob(self, x):
        raise NotImplementedError(
            "the conditional log-density is not available: it needs log p_s(x_s), and the "
            "teacher supplies only its gradient. Run gmmd.ibw.fit with kl_every=0 and select "
            "the model on samples instead (see gmmd/experiments/ibw_conditional.py).")

    # ------------------------------------------------------------------ init + reporting
    def prior_free_conditional(self):
        """``(mean, variance)`` of ``p(x_s | x_t)`` if the marginal ``p_s`` were flat.

        Dropping ``∇ log p_s`` leaves ``N(x_t, (σ_t² - σ_s²) I)``.  It is the natural place to
        start the mixture: it already sits where the conditional's mass is, which a draw from
        ``U([-s, s]^d)`` at d = 196 608 emphatically does not.
        """
        return self.x_t.copy(), float(self.gap)

    def initial_mixture(self, n_components, rng, spread=1.0):
        """``(means, variances)`` for :func:`gmmd.ibw.fit`: draws from the prior-free conditional.

        The components must not start on top of one another or the problem is symmetric in them
        and they can never separate, so each mean is its own draw ``x_t + spread·√gap·z``.
        """
        rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        z = rng.standard_normal((int(n_components), self.dim))
        means = self.x_t[None, :] + spread * np.sqrt(self.gap) * z
        return means, np.full(int(n_components), self.gap)

    def describe(self):
        return dict(kind="ve_bayes_conditional", t=self.t, s=self.s, sigma_t=self.sigma_t,
                    sigma_s=self.sigma_s, gap=self.gap, dim=self.dim,
                    image_shape=list(self.image_shape),
                    definition="grad log p_s(x_s) + (x_t - x_s)/(sigma_t^2 - sigma_s^2); the "
                               "continuous-time reversal's conditional, NOT the discretised "
                               "sampler's transition kernel")
