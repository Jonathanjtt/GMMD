# Experiment 3 continuation — IBW capacity and optimization

Held-out evaluation uses 128 fresh teacher samples per fit. Teacher–teacher baseline (repeated balanced splits): MMD² 0.00004 ± 0.00202; SWD 0.14740 ± 0.01361.

| stage | init | K | BG | iterations | step | seeds | MMD² mean ± SD | SWD mean ± SD | runtime mean (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best_repeat | teacher | 8 | 4 | 400 | 0.01 | 2 | 0.03927 ± 0.00343 | 0.32973 ± 0.00960 | 329.8 |
| bg_sweep | teacher | 4 | 64 | 300 | 0.1 | 1 | 0.17489 ± 0.00000 | 0.71844 ± 0.00000 | 2000.5 |
| early_stopping | teacher | 8 | 4 | 50 | 0.01 | 1 | 0.31198 ± 0.00000 | 0.91461 ± 0.00000 | 41.4 |
| early_stopping | teacher | 8 | 4 | 100 | 0.01 | 1 | 0.22242 ± 0.00000 | 0.74706 ± 0.00000 | 82.2 |
| early_stopping | teacher | 8 | 4 | 200 | 0.01 | 1 | 0.11378 ± 0.00000 | 0.53896 ± 0.00000 | 165.5 |
| early_stopping | teacher | 8 | 4 | 300 | 0.01 | 1 | 0.06404 ± 0.00000 | 0.42161 ± 0.00000 | 248.4 |
| early_stopping | teacher | 8 | 4 | 400 | 0.01 | 1 | 0.03831 ± 0.00000 | 0.35261 ± 0.00000 | 332.9 |
| early_stopping | teacher | 8 | 4 | 600 | 0.01 | 1 | 0.06637 ± 0.00000 | 0.42837 ± 0.00000 | 501.3 |
| optimization | teacher | 8 | 4 | 1000 | 0.01 | 1 | 0.13400 ± 0.00000 | 0.57131 ± 0.00000 | 832.5 |
| optimization | teacher | 8 | 4 | 1000 | 0.03 | 1 | 0.15891 ± 0.00000 | 0.57908 ± 0.00000 | 833.4 |
| optimization | teacher | 8 | 4 | 1000 | 0.1 | 1 | 0.17536 ± 0.00000 | 0.62313 ± 0.00000 | 833.4 |
| screen | standard | 1 | 4 | 300 | 0.1 | 1 | 0.41756 ± 0.00000 | 1.02431 ± 0.00000 | 29.5 |
| screen | standard | 2 | 4 | 300 | 0.1 | 1 | 0.40003 ± 0.00000 | 0.95705 ± 0.00000 | 59.8 |
| screen | standard | 4 | 4 | 300 | 0.1 | 1 | 0.39768 ± 0.00000 | 0.98187 ± 0.00000 | 124.0 |
| screen | standard | 8 | 4 | 300 | 0.1 | 1 | 0.39774 ± 0.00000 | 0.95512 ± 0.00000 | 249.6 |
| screen | standard | 16 | 4 | 300 | 0.1 | 1 | 0.39263 ± 0.00000 | 0.94155 ± 0.00000 | 502.4 |
| screen | teacher | 1 | 4 | 300 | 0.1 | 1 | 0.31076 ± 0.00000 | 0.76454 ± 0.00000 | 29.6 |
| screen | teacher | 2 | 4 | 300 | 0.1 | 1 | 0.16673 ± 0.00000 | 0.64958 ± 0.00000 | 60.1 |
| screen | teacher | 4 | 4 | 300 | 0.1 | 1 | 0.15859 ± 0.00000 | 0.61250 ± 0.00000 | 124.5 |
| screen | teacher | 8 | 4 | 300 | 0.1 | 1 | 0.15443 ± 0.00000 | 0.57234 ± 0.00000 | 249.2 |
| screen | teacher | 16 | 4 | 300 | 0.1 | 1 | 0.16313 ± 0.00000 | 0.63706 ± 0.00000 | 502.1 |
| seed_repeat | teacher | 4 | 16 | 300 | 0.1 | 3 | 0.16891 ± 0.00591 | 0.65591 ± 0.02428 | 498.4 |

Best completed setting: teacher initialization, K=8, BG=4, 400 iterations, step=0.01; MMD²=0.03831.
It remains above MMD² 0.01; follow-up budget and trajectory comparisons are required.

## Diagnostic conclusions

- Standard initialization barely benefits from 16× capacity: MMD² 0.41756 → 0.39263, while fit runtime rises 29.5s → 502.4s.
- Oracle teacher-cluster initialization is the largest improvement: K=1 MMD² 0.31076; best screen K=8 gives 0.15443. Capacity then saturates and K=16 worsens.
- More gradient samples do not help K=4: BG=16 gives MMD² 0.16891 ± 0.00591; BG=64 gives 0.17489, at much higher runtime.
- The best optimization-path point so far is 400 iterations at step 0.01 (MMD² 0.03831). Longer/larger-step optimization can worsen teacher-sample MMD despite lower-noise gradients.
- Along the common-seed step=0.01 path, mean component separation contracts from 48.3 at the best point to 25.6 at 1000 iterations, while MMD² rises to 0.13400. This is objective-driven over-contraction, not component collapse or failure to optimize.
- Repeating the selected 400-step setting across 3 seeds gives MMD² 0.03895 ± 0.00249 and SWD 0.33736 ± 0.01485.
- The remaining gap is most consistent with target/objective mismatch and limited mixture-family capacity: the implementation fixes equal component weights and isotropic pixel-space covariance. More K or Monte-Carlo gradient samples cannot repair either restriction.

## Prioritized next steps (no conditional network)

1. Turn the held-out teacher metric into a validation-only early-stopping rule; confirm the selected iteration on independent evaluation seeds and report its mean ± SD rather than selecting on the test split.
2. Run a weighted-mixture control initialized from teacher clusters. The discovery clusters are strongly imbalanced, while current IBW fixes every component weight to 1/K; this cleanly tests whether weight capacity is the dominant residual gap.
3. Separate objective from family mismatch with a direct teacher-sample fit/control, then compare it to score-based IBW under the same K, samples, and evaluation seeds.
4. If weights are insufficient, add a memory-feasible covariance extension (channelwise/diagonal or low-rank-plus-isotropic), starting at K=4 or K=8. Full pixel covariance is infeasible at 256² resolution.
5. Only after those controls, revisit step schedules or larger K/BG. Stop increasing BG now: the measured cost scales nearly linearly without an accuracy gain.

All fit checkpoints, convergence traces, resolved settings, and per-run metrics are retained.
