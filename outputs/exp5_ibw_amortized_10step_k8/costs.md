| item | category | units | NFE | GPU-hours |
|---|---|---|---|---|
| anchor trajectories (train; teacher fp32) | training data | 64 | 1.15e+05 | 2.36 |
| teacher endpoints (train, fp32, ablation data) | ablation | 640 | 5.12e+05 | 5.82 |
| IBW fits train/frozen (TF32; prior_free init) | training data | 640 | 5.14e+05 | 3.14 |
| anchor trajectories (val; teacher fp32) | selection (one-off: P0/P1 on val) | 4 | 7.2e+03 | 0.25 |
| teacher reference rollouts (val, fp32) | selection (one-off: P0/P1 on val) | 40 | 5.12e+05 | 5.33 |
| IBW fits val/frozen_r1 (TF32; prior_free init) | evaluation only | 40 | 3.21e+04 | 0.26 |
| IBW fits val/p0_c0p2 (TF32; prior_free init) | selection (one-off: P0/P1 on val) | 6 | 7.68e+04 | 0.24 |
| IBW fits val/p0_c0p5 (TF32; prior_free init) | selection (one-off: P0/P1 on val) | 6 | 7.68e+04 | 0.24 |
| IBW fits val/p0_c1p0 (TF32; prior_free init) | selection (one-off: P0/P1 on val) | 6 | 7.68e+04 | 0.26 |
| IBW fits val/principal_r1 (TF32; principal-direction init fp32) | selection (one-off: P0/P1 on val) | 40 | 3.88e+05 | 1.25 |
| IBW fits val/principal_r3 (TF32; principal-direction init fp32) | selection (one-off: P0/P1 on val) | 40 | 3.88e+05 | 1.31 |
| IBW fits val/principal_r3_vgap (TF32; principal-direction init fp32) | selection (one-off: P0/P1 on val) | 40 | 3.88e+05 | 1.22 |
| IBW fits val/prior_free (TF32; prior_free init) | selection (one-off: P0/P1 on val) | 40 | 3.84e+05 | 2.29 |
| IBW fits val/prior_free_fine (TF32; prior_free init) | selection (one-off: P0/P1 on val) | 40 | 6.4e+04 | 0.43 |
| IBW fits val/random_far (TF32; random_far init) | selection (one-off: P0/P1 on val) | 40 | 3.84e+05 | 1.21 |
| anchor trajectories (test; teacher fp32) | evaluation only | 8 | 1.44e+04 | 0.31 |
| teacher reference rollouts (test, fp32) | evaluation only | 80 | 1.02e+06 | 15.21 |
| IBW fits test/frozen (TF32; prior_free init) | evaluation only | 80 | 6.43e+04 | 0.51 |
| IBW fits test/frozen_fp32 (TF32; prior_free init) | evaluation only | 20 | 1.61e+04 | 0.30 |
| IBW fits test/frozen_r1 (TF32; prior_free init) | evaluation only | 80 | 6.43e+04 | 0.44 |
| anchor states (refine; student rollouts, TF32) | refinement data (student-rollout anchors + fits) | 32 | 0 | 0.00 |
| IBW fits refine/frozen (TF32; prior_free init) | refinement data (student-rollout anchors + fits) | 288 | 2.31e+05 | 0.71 |
| student training: ablation_endpoints (TF32, 12000 steps) | ablation | 12000 | 0 | 1.74 |
| student training: main (TF32, 12000 steps) | student training | 12000 | 0 | 1.81 |
| student training: pilot (TF32, 3000 steps) | other student | 3000 | 0 | 0.77 |
| student training: refined (TF32, 4000 steps) | other student | 4000 | 0 | 0.61 |

| category | GPU-hours |
|---|---|
| training data | 5.50 |
| ablation | 7.56 |
| selection (one-off: P0/P1 on val) | 14.03 |
| evaluation only | 17.03 |
| refinement data (student-rollout anchors + fits) | 0.71 |
| student training | 1.81 |
| other student | 1.38 |

| uncached generation (x_T → image, one batch after warm-up, synchronised) | precision | batch | s / image | images / GPU-hour | peak GB |
|---|---|---|---|---|---|
| coarse:ancestral:10 | fp32 | 16 | 0.4807 | 7489 | 30.4 |
| coarse:ancestral:10 | fp32 | 64 | 0.3751 | 9598 | 48.3 |
| coarse:ancestral:10 | tf32 | 16 | 0.1364 | 26395 | 5.8 |
| coarse:ancestral:10 | tf32 | 64 | 0.1076 | 33471 | 22.5 |
| coarse:ancestral:20 | fp32 | 16 | 0.9627 | 3740 | 30.4 |
| coarse:ancestral:20 | tf32 | 16 | 0.2575 | 13982 | 5.8 |
| coarse:ancestral:20 | tf32 | 64 | 0.2258 | 15946 | 22.5 |
| coarse:ancestral:50 | fp32 | 16 | 2.4006 | 1500 | 30.4 |
| coarse:ancestral:50 | tf32 | 16 | 0.6106 | 5896 | 5.8 |
| coarse:ancestral:50 | tf32 | 64 | 0.5578 | 6453 | 22.5 |
| student:ablation_endpoints | fp32 | 16 | 0.5040 | 7143 | 30.5 |
| student:ablation_endpoints | tf32 | 16 | 0.1342 | 26822 | 6.0 |
| student:ablation_endpoints | tf32 | 64 | 0.1162 | 30976 | 23.0 |
| student:main | fp32 | 16 | 0.4776 | 7537 | 30.5 |
| student:main | fp32 | 64 | 0.4187 | 8598 | 48.6 |
| student:main | tf32 | 16 | 0.1095 | 32888 | 6.0 |
| student:main | tf32 | 64 | 0.1147 | 31383 | 23.0 |
| teacher | fp32 | 16 | 95.6318 | 38 | 30.4 |
| teacher | tf32 | 16 | 24.0867 | 149 | 5.9 |

Break-even (fp32_batch16, student main vs teacher): offline 7.3 GPU-h (training data + student training; students main) / saving 95.15 s per image (speed-up 200x) = 276 images (807 including the one-off selection)

Break-even (fp32_batch64, student main vs teacher): offline 7.3 GPU-h (training data + student training; students main) / saving 95.21 s per image (speed-up 228x) = 276 images (807 including the one-off selection)

Break-even (tf32_batch16, student main vs teacher): offline 7.3 GPU-h (training data + student training; students main) / saving 23.98 s per image (speed-up 220x) = 1097 images (3203 including the one-off selection)

Break-even (tf32_batch64, student main vs teacher): offline 7.3 GPU-h (training data + student training; students main) / saving 23.97 s per image (speed-up 210x) = 1097 images (3204 including the one-off selection)
