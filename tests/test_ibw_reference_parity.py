"""``gmmd.ibw`` reproduces the authors' reference implementation exactly.

:mod:`gmmd.ibw` is written from the equations of Petit-Talamon, Lambert & Korba (2025), not
copied from their code -- their repository https://github.com/margueritetalamon/VI-MIG carries
no licence file, so nothing of it is redistributed here.  This test closes that gap: it fetches
their sources at a pinned commit *at test time*, into a temporary directory, and checks that
the two implementations agree to machine precision on

* the isotropic-mixture score and log-density,
* the Proposition 3.1 Monte-Carlo gradients, under one fixed noise block,
* all four update formulas: (GD) on the means, (12) IBW, (13) MD, (14) NGD.

It also asserts that the update expressions in their ``src/optim.py`` are still literally the
ones implemented here, so an upstream change to the scheme is caught rather than silently
diverged from.

Marked ``network``; skipped offline.  Run with ``pytest -m network``.
"""

from __future__ import annotations

import importlib.util
import re
import urllib.request

import numpy as np
import pytest

from gmmd import ibw
from gmmd.vi import isotropic_mixture_target

pytestmark = pytest.mark.network

UPSTREAM = "https://github.com/margueritetalamon/VI-MIG"
COMMIT = "a3d3fd14ff5588037f7660830782544010e1ea21"
RAW = f"https://raw.githubusercontent.com/margueritetalamon/VI-MIG/{COMMIT}/"

# Their src/optim.py, VI_GMM.optimize: the four update lines this module reimplements.
# Whitespace-insensitive, so reformatting upstream does not fail the test but a change to the
# scheme does.
REFERENCE_UPDATES = {
    "GD means": r"new_means\s*=\s*self\.vgmm\.means\s*-\s*learning_rate\s*\*\s*self\.vgmm\.n_components\s*\*\s*grad_means",
    "(12) IBW": r"new_epsilons\s*=\s*\(1\s*-\s*\(2\*self\.vgmm\.n_components\*learning_rate/self\.dim\)\s*\*\s*grad_covs\)\*\*2\s*\*\s*self\.vgmm\.epsilons",
    "(13) MD": r"new_epsilons\s*=\s*self\.vgmm\.epsilons\s*\*\s*np\.exp\(-\(2\*self\.vgmm\.n_components\*learning_rate/self\.dim\)\s*\*\s*grad_covs\s*\)",
    "(14) NGD eps": r"inv_new_epsilons\s*=\s*\(1/self\.vgmm\.epsilons\)\s*\+\s*\(2\s*\*\s*self\.vgmm\.n_components\s*\*\s*learning_rate\s*\*\s*grad_covs\s*/\s*self\.dim\)",
    "(14) NGD means": r"new_means\s*=\s*self\.vgmm\.means\s*-\s*new_epsilons\[:,\s*None\]\s*\*\s*self\.vgmm\.n_components\s*\*\s*learning_rate\s*\*\s*grad_means",
}

# The fixed problem both implementations are evaluated on.
TARGET_MEANS = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 4.0]])
TARGET_VARS = np.array([0.5, 1.2, 0.8])
TARGET_WEIGHTS = np.array([0.2, 0.5, 0.3])
Q_MEANS = np.array([[1.0, 0.5], [-0.5, -1.0], [2.0, -2.0], [0.0, 0.0]])
Q_VARS = np.array([0.7, 1.3, 0.4, 2.0])
STEP = 0.05


def _fetch(rel):
    try:
        with urllib.request.urlopen(RAW + rel, timeout=30) as r:
            return r.read().decode()
    except Exception as e:  # offline, DNS, proxy, upstream gone
        pytest.skip(f"cannot reach {UPSTREAM}: {e}")


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Their ``src/gmm.py`` imported as a module from a temp dir (never written to this repo)."""
    pytest.importorskip("einops", reason="the reference implementation imports einops")
    pytest.importorskip("torch", reason="the reference implementation imports torch")
    d = tmp_path_factory.mktemp("vi_mig_reference")
    path = d / "reference_gmm.py"
    path.write_text(_fetch("src/gmm.py"))
    spec = importlib.util.spec_from_file_location("vi_mig_reference_gmm", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"the reference src/gmm.py no longer imports standalone: {e}")
    return mod


@pytest.fixture(scope="module")
def pair(reference):
    """``(their_target, their_mixture, our_target, our_mixture)`` on the same fixed problem."""
    their_t = reference.GMM(variational=False, mode="iso", weights=TARGET_WEIGHTS,
                            means=TARGET_MEANS,
                            covs=np.stack([v * np.eye(2) for v in TARGET_VARS]))
    their_q = reference.IGMM(means=Q_MEANS, covs=np.stack([v * np.eye(2) for v in Q_VARS]))
    our_t = isotropic_mixture_target(TARGET_MEANS, TARGET_VARS, TARGET_WEIGHTS)
    our_q = ibw.IsotropicMixture(Q_MEANS, Q_VARS)
    return their_t, their_q, our_t, our_q


def test_the_reference_family_is_the_one_we_implement(pair):
    their_t, their_q, _, our_q = pair
    assert np.allclose(their_q.epsilons, our_q.variances)
    assert np.allclose(their_q.weights, our_q.weights)       # uniform 1/N
    assert their_q.n_components == our_q.n_components and their_q.dim == our_q.dim


def test_scores_and_log_densities_agree(pair):
    their_t, their_q, our_t, our_q = pair
    x = np.random.RandomState(7).randn(50, 2) * 3
    assert np.abs(their_t.gradient_log_density(x) - our_t.score(x)).max() < 1e-12
    assert np.abs(their_q.gradient_log_density(x) - our_q.score(x)).max() < 1e-12
    assert np.abs(np.asarray(their_t.log_prob(x)).ravel() - our_t.log_prob(x)).max() < 1e-12


@pytest.mark.parametrize("B", [1, 9, 64])
def test_proposition_31_gradients_agree(pair, B):
    their_t, their_q, our_t, our_q = pair
    noise = np.random.RandomState(B).randn(B, 2)
    their_gm, their_ge = their_q.compute_grads(their_t, noise=noise, B=B)
    our_gm, our_ge = ibw.mixture_gradients(our_q, our_t, noise)
    assert np.abs(their_gm - our_gm).max() < 1e-12
    assert np.abs(their_ge - our_ge).max() < 1e-12


def test_all_four_update_formulas_agree(pair):
    their_t, their_q, our_t, our_q = pair
    noise = np.random.RandomState(0).randn(9, 2)
    gm, ge = their_q.compute_grads(their_t, noise=noise, B=9)
    N, d = their_q.n_components, their_q.dim

    # transcriptions of src/optim.py's update lines, evaluated on THEIR gradients
    their_gd_means = their_q.means - STEP * N * gm
    their_ibw_eps = (1 - (2 * N * STEP / d) * ge) ** 2 * their_q.epsilons
    their_md_eps = their_q.epsilons * np.exp(-(2 * N * STEP / d) * ge)
    their_ngd_eps = ((1 / their_q.epsilons) + (2 * N * STEP * ge / d)) ** -1
    their_ngd_means = their_q.means - their_ngd_eps[:, None] * N * STEP * gm

    assert np.abs(their_gd_means - (our_q.means - STEP * N * gm)).max() < 1e-12
    assert np.abs(their_ibw_eps - ibw.ibw_variance_update(Q_VARS, ge, N, d, STEP)).max() < 1e-12
    assert np.abs(their_md_eps - ibw.md_variance_update(Q_VARS, ge, N, d, STEP)).max() < 1e-12
    ngd_m, ngd_v, ok = ibw.ngd_update(our_q.means, Q_VARS, gm, ge, N, d, STEP)
    assert ok
    assert np.abs(their_ngd_eps - ngd_v).max() < 1e-12
    assert np.abs(their_ngd_means - ngd_m).max() < 1e-12


def test_upstream_still_uses_the_scheme_we_implemented():
    """If upstream changes an update rule, fail loudly instead of silently diverging."""
    src = _fetch("src/optim.py")
    missing = [name for name, pat in REFERENCE_UPDATES.items() if not re.search(pat, src)]
    assert not missing, (f"the reference implementation at {COMMIT[:8]} no longer matches "
                         f"gmmd/ibw.py for: {missing}. Re-read src/optim.py before trusting "
                         "this module's updates.")
