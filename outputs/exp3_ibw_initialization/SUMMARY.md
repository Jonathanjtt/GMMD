# Experiment 3 — IBW initialisation diagnostic

Target: continuous-time VE Bayes conditional for the canonical H1 0.60→0.45 state.
Teacher-cluster initialisation is an oracle diagnostic, not a practical method.

| condition | mean DINOv2 MMD² | mean SWD | mean normalized separation |
|---|---:|---:|---:|
| K=1, standard | 0.43958 | 0.95738 | nan |
| K=2, standard | 0.44129 | 1.01870 | 2.486 |
| K=4, standard | 0.43610 | 0.96919 | 2.844 |
| K=2, deliberately spread | 0.42516 | 0.99196 | 3.550 |
| K=2, teacher-cluster oracle | 0.21542 | 0.72155 | 9.512 |

See `numerical/results.json` for raw distances, collapse flags, and trajectories.
