# Experiment 2 — controlled H1 test

Target: equal-weight 2D Gaussian mixture with identity component covariance.
`direct_full_covariance` is the strong sample-fit reference; `ibw_isotropic` is the existing IBW family.

| separation | fit family | mean SWD advantage (K=1 − K=2) |
|---:|---|---:|
| 0 | direct_full_covariance | -0.00646 |
| 0 | ibw_isotropic | -0.00015 |
| 1 | direct_full_covariance | -0.00456 |
| 1 | ibw_isotropic | 0.00734 |
| 2 | direct_full_covariance | 0.01150 |
| 2 | ibw_isotropic | 0.08991 |
| 4 | direct_full_covariance | 0.15739 |
| 4 | ibw_isotropic | 0.47852 |
| 6 | direct_full_covariance | 0.40579 |
| 6 | ibw_isotropic | 1.83177 |

Positive advantage means the two-component approximation is better.
