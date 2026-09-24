"""The multimodality diagnostics on clouds whose answer is known."""

import numpy as np
import pytest

from gmmd import diagnostics as D

KW = dict(pca_dims=8, max_k=3, n_boot=60, n_null=30, seed=0)


@pytest.fixture(scope="module")
def clouds():
    rng = np.random.RandomState(0)
    scale = np.linspace(1.5, 0.5, 40)                      # anisotropic, largest axis first
    uni = rng.randn(64, 40) * scale
    a = rng.randn(32, 40) * scale + np.r_[6.0, np.zeros(39)]   # modes 12 apart on a 1.5-std axis
    b = rng.randn(32, 40) * scale - np.r_[6.0, np.zeros(39)]
    return dict(unimodal=uni, bimodal=np.concatenate([a, b]), degenerate=np.ones((6, 40)))


def test_unimodal_cloud_is_not_declared_multimodal(clouds):
    r = D.analyze(clouds["unimodal"], **KW)
    assert not r["degenerate"]
    assert r["gmm"]["best_k_bic"] == 1
    assert r["lrt"]["p_value"] > 0.2
    assert r["stability"]["p_silhouette"] > 0.05
    assert r["spread"]["pairwise_min"] > 0


def test_bimodal_cloud_is_detected(clouds):
    r = D.analyze(clouds["bimodal"], **KW)
    assert r["gmm"]["best_k_bic"] >= 2 and r["gmm"]["best_k_cv"] >= 2
    assert r["lrt"]["p_value"] < 0.05
    assert r["stability"]["ari_mean"] > 0.9 and r["stability"]["p_ari"] < 0.1
    assert r["stability"]["silhouette"] > r["stability"]["null_silhouette_q95"]
    assert sorted(r["stability"]["cluster_sizes"]) == [32, 32]


def test_degenerate_cloud_is_flagged_not_analysed(clouds):
    r = D.analyze(clouds["degenerate"], **KW)
    assert r["degenerate"] and "gmm" not in r
    assert r["spread"]["pairwise_max"] == 0.0
    assert any("degenerate" in ln for ln in D.summarize(r, "x"))


def test_summary_lines_are_strings(clouds):
    r = D.analyze(clouds["unimodal"], **KW)
    lines = D.summarize(r, "u")
    assert len(lines) >= 4 and all(isinstance(ln, str) for ln in lines)
