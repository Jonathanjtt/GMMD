"""Fast structural checks for the three focused follow-up experiments."""

import numpy as np

from gmmd.experiments.ibw_initialization import _component_stats
from gmmd.experiments.ibw_capacity_sweep import _row, expand_run_grid
from gmmd.experiments.toy_h1 import _direct_fits, _target_parameters
from gmmd.experiments.transition_multimodality_combined import aggregate


def test_exp3_collapse_rule_keeps_raw_and_normalized_distances():
    close = _component_stats(np.array([[0.0, 0.0], [1.0, 0.0]]), np.ones(2), 1.0)
    apart = _component_stats(np.array([[0.0, 0.0], [4.0, 0.0]]), np.ones(2), 1.0)
    assert close["pairwise_distance_min"] == 1.0
    assert close["normalized_separation_min"] == 0.5 and close["collapsed"]
    assert close["effective_separated_components"] == 1
    assert apart["normalized_separation_min"] == 2.0 and not apart["collapsed"]
    assert apart["effective_separated_components"] == 2


def test_exp2_direct_fits_are_one_and_two_component_models():
    means, _, _ = _target_parameters(4.0)
    rng = np.random.default_rng(0)
    train = np.concatenate([rng.normal(means[0], 1.0, size=(100, 2)),
                            rng.normal(means[1], 1.0, size=(100, 2))])
    q1, q2 = _direct_fits(train, seed=0)
    assert q1["means"].shape == (1, 2) and q1["covariances"].shape == (1, 2, 2)
    assert q2["means"].shape == (2, 2) and q2["covariances"].shape == (2, 2, 2)
    assert np.isclose(q2["weights"].sum(), 1.0)


def test_exp3_capacity_grid_expands_and_deduplicates():
    specs = expand_run_grid([
        dict(stage="screen", init="standard", k=[1, 2], bg=[4, 16], seeds=[11],
             iterations=[300], step_sizes=[0.1]),
        dict(stage="screen", init="standard", k=[2], bg=[4], seeds=[11],
             iterations=[300], step_sizes=[0.1]),
    ])
    assert len(specs) == 4
    assert specs[0] == dict(stage="screen", init="standard", k=1, bg=4, seed=11,
                            iterations=300, step_size=0.1)
    assert specs[-1]["k"] == 2 and specs[-1]["bg"] == 16


def test_exp3_capacity_grid_rejects_invalid_budget():
    import pytest

    with pytest.raises(ValueError, match="invalid run specification"):
        expand_run_grid([dict(k=[2], bg=[0], seeds=[11], iterations=[300], step_sizes=[0.1])])


def test_exp3_capacity_row_combines_fit_and_evaluation_metrics():
    spec = dict(stage="screen", init="standard", k=1, bg=4, seed=11,
                iterations=3, step_size=0.1)
    fit = dict(means=np.array([[1.0, 2.0]]), variances=np.array([1.5]),
               grad_norms=np.array([3.0, 2.0, 1.0]), runtime_s=0.5,
               n_iterations_run=3, stopped_early=None)
    quality = dict(n_eval=128, eval_seed=1, mmd2=0.2, mmd_p_value=0.01,
                   mmd_null_q95=0.02, mmd_bandwidth=1.0, sliced_wasserstein=0.3,
                   swd_projection_std=0.1)
    row = _row(spec, fit, np.zeros((1, 2)), np.ones(1), quality,
               dict(collapse_threshold=1.0, batch_size=16))
    assert row["n_eval"] == 128 and row["mmd2"] == 0.2
    assert row["score_samples"] == 12 and row["network_batches_estimate"] == 3


def test_combined_transition_summary_counts_k2_evidence():
    rows = [
        dict(s=0.58, delta_t=0.02, delta_bic_2_minus_1=-2.0, preferred_k_bic=2),
        dict(s=0.58, delta_t=0.02, delta_bic_2_minus_1=4.0, preferred_k_bic=1),
        dict(s=0.45, delta_t=0.15, delta_bic_2_minus_1=-8.0, preferred_k_bic=3),
        dict(s=0.45, delta_t=0.15, delta_bic_2_minus_1=-1.0, preferred_k_bic=2),
    ]
    summary = aggregate(rows)
    assert [x["k2_favored_count"] for x in summary] == [1, 2]
    assert [x["k2_favored_fraction"] for x in summary] == [0.5, 1.0]
    assert summary[1]["preferred_mixture_count"] == 2
