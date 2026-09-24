"""Targets for IBW, and the reason the teacher conditional is not one of them yet.

:mod:`gmmd.ibw` fits a mixture of isotropic Gaussians to any object exposing

    score(x):    (M, d) -> (M, d)    grad_x log pi(x)      -- drives every update
    log_prob(x): (M, d) -> (M,)      log pi(x) + const     -- only the reverse-KL diagnostic

This module supplies the analytic targets used to test that optimiser, and states the
interface the *teacher* conditional would have to satisfy.

Why the teacher conditional is blocked
--------------------------------------
The object we ultimately want to approximate is ``p_theta(x_s | x_t)``, so reverse-KL VI needs

    grad_{x_s} log p_theta(x_s | x_t).

What the teacher actually provides is the MARGINAL score ``grad log p_s(x_s)`` (the score
network) plus a *sampler*: the transition ``x_t -> x_s`` is a composition of per-step Gaussian
kernels over the discretised reverse SDE, and that composition has no closed-form density or
score for more than one step.  (A single reverse-diffusion predictor step is Gaussian, and its
score is trivial; the 299-step transition of the H1 experiment is not.)

There is a closed form for a *different* object.  For the exact continuous-time time reversal
under a perfect score, Bayes' rule on the VE forward kernel gives

    grad_{x_s} log p(x_s | x_t) = grad log p_s(x_s) + (x_t - x_s) / (sigma_t^2 - sigma_s^2),

which needs only the network and the forward kernel.  But that is the continuous-time reversal
under the *learned marginal* score, not the kernel of the discretised sampler that generates
the samples we are characterising -- and it is certainly not the kernel of the
predictor-corrector sampler, whose Langevin steps re-target the marginal at every level.

Adopting it, or a sample-based objective, or density-ratio estimation, or distillation, is a
modelling decision with different consequences for what the H1 result would mean.  So
:class:`TeacherConditionalTarget` raises instead of quietly picking one.  Record the decision
here before running IBW against the teacher.
"""

from __future__ import annotations

import numpy as np

__all__ = ["AnalyticTarget", "isotropic_mixture_target", "TeacherConditionalTarget"]


class AnalyticTarget:
    """A target defined by explicit ``score`` / ``log_prob`` callables."""

    def __init__(self, dim, score, log_prob, name="analytic"):
        self.dim = int(dim)
        self._score = score
        self._log_prob = log_prob
        self.name = name

    def score(self, x):
        return np.asarray(self._score(np.asarray(x, dtype=float)), dtype=float)

    def log_prob(self, x):
        return np.asarray(self._log_prob(np.asarray(x, dtype=float)), dtype=float)


def isotropic_mixture_target(means, variances, weights=None, name="isotropic_mixture"):
    """A mixture-of-isotropic-Gaussians target with exact score and log-density.

    The one target whose answer is known in closed form, so a fit can be checked rather than
    eyeballed: with ``N`` components placed on ``N`` well-separated modes of equal variance,
    the exact minimiser of the reverse KL over this family is the target itself.
    """
    from scipy.special import logsumexp, softmax

    means = np.asarray(means, dtype=float)
    if means.ndim != 2:
        raise ValueError(f"means must be (K, d), got {means.shape}")
    var = np.asarray(variances, dtype=float).reshape(-1)
    K, d = means.shape
    if var.shape[0] != K:
        raise ValueError(f"{K} means but {var.shape[0]} variances")
    logw = np.log(np.full(K, 1.0 / K) if weights is None else
                  np.asarray(weights, dtype=float) / np.sum(weights))

    def _log_components(x):
        diff = x[:, None, :] - means[None, :, :]
        quad = np.einsum("mkd,mkd->mk", diff, diff) / var[None, :]
        return logw[None, :] - 0.5 * d * (np.log(2 * np.pi) + np.log(var))[None, :] - 0.5 * quad

    def log_prob(x):
        return logsumexp(_log_components(x), axis=1)

    def score(x):
        r = softmax(_log_components(x), axis=1)
        diff = x[:, None, :] - means[None, :, :]
        return -np.einsum("mk,mkd->md", r, diff / var[None, :, None])

    t = AnalyticTarget(d, score, log_prob, name=name)
    t.means, t.variances, t.weights = means, var, np.exp(logw)
    return t


class TeacherConditionalTarget:
    """``p_theta(x_s | x_t)`` as an IBW target -- BLOCKED on the choice of target quantity.

    See the module docstring.  Instantiating this raises, so that no experiment can silently
    substitute the marginal score, the Bayes-rule surrogate, or a sample-based objective for
    the reverse-KL target the question actually names.
    """

    def __init__(self, teacher, x_t, t, s):
        raise NotImplementedError(
            "IBW needs grad_{x_s} log p_theta(x_s | x_t); the teacher provides only the marginal "
            "score grad log p_s(x_s) and a sampler. Candidate substitutes (the Bayes-rule "
            "surrogate score_theta(x_s, s) + (x_t - x_s) / (sigma_t^2 - sigma_s^2); a "
            "sample-based objective; density-ratio or conditional-score estimation; "
            "distillation) are a modelling decision -- record the choice in gmmd/vi.py before "
            "running IBW against the teacher.")
