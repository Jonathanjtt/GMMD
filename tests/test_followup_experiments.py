"""Fast structural checks for the three focused follow-up experiments."""

import numpy as np

from gmmd.experiments.ibw_initialization import _component_stats
from gmmd.experiments.toy_h1 import _direct_fits, _target_parameters


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
