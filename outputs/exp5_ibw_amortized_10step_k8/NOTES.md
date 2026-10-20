# exp5 working notes (findings to carry into REPORT.md)

## 2026-10-19 P0 (prior-free K = 8, c ∈ {0.2, 0.5, 1.0}, val trajectories 0–1, intervals 1/4/7)

* Convergence: with γ_j = c·v_anc,j every fit reaches a gradient plateau (|∇_m F| down 25–1000×) by
  50–100 iterations; variances settle at 1.00–1.05 × v_anc. The pre-registered drift test fails at
  interval 7 only because the Monte-Carlo noise floor (B = 4) is higher there (drift 0.03–0.10 vs
  tolerance 0.02) — a secondary, labelled gradient-plateau test classifies all 18 fits stationary.
* Prior-free components do not collapse at mid/low noise (n_eff = 8, median separation 2.3–3 σ) but
  do at high noise (interval 1: n_eff = 1).
* Converged stage-A DINOv2 excess ≈ 0.7 (j = 1), 0.15–0.25 (j = 4), 0.5–1.0 (j = 7), identical
  across c; early transients (larger variance) score better (c = 0.2 at 50 it: j = 1 0.13–0.17).
* Dispersion (mean pairwise DINOv2 distance, IBW / teacher; teacher-vs-teacher control 0.97–1.08):
  j = 4: 0.65–0.76, j = 7: 0.61–0.76 → IBW samples under-dispersed in content; pixel dispersion at
  j = 7 0.64–0.71. j = 1: 1.8–2.3 (inverted — see next point).
* Images (`figures/diag/denoised_teacher_vs_ibw_j{1,4}.png`, Tweedie-denoised, val traj 0):
  - j = 4: teacher samples are diverse sharp faces (hats, hair, poses); IBW samples are near-identical
    blurry faces → the isotropic uniform-weight family from the prior-free start sits on the
    conditional mean in content directions.
  - j = 1: denoising the TEACHER's own states leaves heavy colour noise (denoised range −5.1…5.6,
    impossible for an exact posterior mean), whereas IBW samples denoise to clean faces (range
    −0.67…1.78). Raw, the two sets are almost identical in second moments (per-pixel distance to the
    IBW mean 41.0 vs 40.2, v_anc = 40.2²). [Superseded interpretation, see the score-norm check
    below: this is NOT excess total noise in the teacher's states.] DINOv2 clamps the saturated
    teacher images, so its dispersion there is an artifact; the pixel metric does not see
    high-frequency structure after 8× pooling.
  - Implication: the teacher's states differ from IBW samples in directions the network's denoiser
    amplifies, so students trained on teacher-trajectory anchors will see IBW-like states at
    inference → rollout shift is expected; refinement on student-rollout anchors addresses it.
* Score-norm check (`figures/diag/score_norm_ratio_val0.json`, val traj 0, 16 samples):
  σ_s² ‖s_θ(y, s)‖² / d = 0.9996–0.9999 for BOTH teacher states and prior-free IBW samples at
  j = 0–2, so the teacher's states carry the expected total noise energy; the difference that makes
  their Tweedie-denoised images noisy lives in a low-energy subspace where the network's denoiser
  has a large gain (consistent with the teacher's anisotropic conditional covariance, Rayleigh
  quotients up to 2 × gap, versus IBW's isotropic v_anc). For teacher states the ratio falls at low
  noise (j = 7: 0.965, j = 8: 0.872, j = 9 [σ = 0.01]: 0.171). Conclusion for evaluation: at the
  high-noise intervals the DINOv2-on-Tweedie metric (and its dispersion) is dominated by this
  effect plus clamping; pixel metrics and the end-to-end FID carry the interpretable signal there.

## 2026-10-19 P1 (5 teacher-free initialisations x 40 val pairs, 300 iterations, c = 0.5) -> selection

* Pre-registered selection (`selection.json`, frozen 15:47 UTC): **prior_free @ 25 iterations**.
  Mean DINOv2 excess over the 40 val pairs at the best snapshot: prior_free 0.391 (25 it);
  principal_r3_vgap 0.550, principal_r3 0.575, principal_r1 0.578, random_far 0.579 (all at 25 it).
  Paired bootstrap of (variant − prior_free): +0.16 … +0.19, every 95 % CI excludes 0.
  Figure: `figures/fig_p1_stage_a_vs_iterations.{png,pdf}`; tables `fitsummary_val_*.md`.
* At 300 iterations every initialisation reaches the same place (mean excess 0.617–0.625; n_eff 6.6;
  v/v_anc 1.00 at j0 → 1.09 at j8, 1.23 at j9): the IBW fixed point does not depend on the start,
  including random_far (initial separations ≈ 313 σ collapse to 0.5–4 σ).
* The selected snapshot is a TRANSIENT, not a converged fit (drift test: 0/4 converged at 25 it by
  construction). Mechanism: the prior-free start puts every component mean at `x + sqrt(gap) z` with
  variance gap; the mean update contracts each direction at rate ≈ γ·precision, i.e. c = 0.5 per
  iteration in stiff (noise-like, precision ≈ 1/v_anc) directions, but only ≈ c·v_anc/gap ≈ 0.06 in
  soft content directions (precision ≈ 1/gap). After 25 iterations the stiff directions have
  converged while ≈ 20 % of the random content-direction offsets remain → the mixture keeps content
  dispersion the reverse-KL fixed point removes. The principal/random_far starts lack these offsets
  (they start at the Tweedie mean) and gain little from early stopping.
* Where it helps (prior_free 25 it vs 300 it, DINOv2 excess): j3 0.17/0.20, j4 0.13/0.22, j5 0.21/0.36,
  j6 0.30/0.53, j7 0.22/0.75, j8 0.46/1.01, j9 0.31/1.04; j0–j2 identical (≈ 0.7, the teacher-state
  denoising artifact). Pixel excess is slightly WORSE at 25 it in j7–j9 (0.09/0.05, 0.14/0.06,
  0.10/0.09) while pixel dispersion is closer to the teacher (j8 0.70 vs 0.58; j9 0.62 vs 0.41).
* Images (`figures/diag/samples_val000_j{4,7,9}_prior_free_it25-50-300.png`): at j4 the teacher's
  200-step kernel produces visibly different faces (pose, expression, hat), every IBW snapshot gives
  near-identical blurry faces — the isotropic K = 8 family captures little of the content spread at
  mid noise, early stopping or not; at j7 and j9 all sets are visually close (differences are fine
  texture), so the large DINOv2 excesses there are small systematic differences measured against a
  tiny within-set spread (median-heuristic bandwidth).
* Process fix: GPU 0 started the stage-A evaluation of principal_r3_vgap while GPU 1 was still
  fitting its last three intervals (12 pairs unevaluated → the selection refused to freeze);
  `run/exp5_pilot_p1.sh` now waits for the variant's fit lock, `run/exp5_pilot_p1_finish.sh`
  evaluated the missing 12 pairs (no fit was redone).

## Post-hoc additions after the freeze (labelled as such everywhere)

* `prior_free_fine` (val, fresh seeds): snapshots 5–50 iterations — the frozen 25 was the lowest
  point of the pre-registered grid, so this checks whether the grid bracketed the optimum.
  Sensitivity analysis only; `selection.json` is not overwritten.
* `frozen_r1` replicate fits (fresh init + noise seeds) at val and test anchors: the IBW-vs-IBW
  floor for stage B. The early-stopped targets depend on the init seed, so two sample sets of ONE
  fit understate what a perfect amortisation can reach.
* IBW chain (`gmmd/exp5/oracle.py`, `generate --method ibwchain`): online IBW with the frozen
  configuration at every transition from the same x_T — a diagnostic ceiling for the amortised
  sampler (no amortisation error, no rollout shift), 8,000 TF32 score evaluations per image.

## 2026-10-19 Post-hoc fine iteration grid (prior_free_fine, fresh seeds, val 40 pairs) — sensitivity only

* Reproducibility: the fresh-seed fits reproduce P1 at the shared snapshots (25 it: 0.393 vs 0.391;
  50 it: 0.570 vs 0.566; per-pair correlation 0.995) — the stage-A excess is a property of the anchor,
  not of the init seed.
* Mean DINOv2 / pixel excess vs iterations: 5: 1.08 / 0.385; 10: 0.99 / 0.198; 15: 0.64 / 0.114;
  **20: 0.354 / 0.069; 25: 0.393 / 0.042; 30: 0.406 / 0.027**; 40: 0.50 / 0.019; 50: 0.57 / 0.020.
  Rule (2) on this grid would pick 20, but 25 and 30 are not separable from 20 (paired CIs of the
  difference [−0.031, +0.113] and [−0.027, +0.130]), while 5–15 and 40–50 are (CIs exclude 0).
  → the pre-registered grid bracketed a broad optimum; the frozen 25 stays (it has the lower pixel
  excess, and the 20-it DINOv2 advantage comes mostly from j0–j1, where the Tweedie-denoised DINOv2
  metric rewards residual noise — the teacher-state artifact). Figure `figures/fig_posthoc_ibw_iterations`.
* Interpretation: the useful IBW iterate lies between under-contraction (≤ 15 it: too much variance,
  pixel excess 0.1–0.4) and over-contraction (≥ 40 it: content dispersion removed by the reverse-KL
  flow). Consistent with P0: c = 0.2 was best at 50 it, c = 0.5 at 25 it (c × iterations ≈ 10–12).

## 2026-10-19 P2 pilot (8 train trajectories, 3,000 steps) — training on IBW targets HURTS end to end

* Training: val NLL 1.4294 → 1.4111 (plateau from ~2,500 steps); forward KL/dim falls to ≈ 0 at j0–j6,
  0.004 (j7), 0.009 (j8); mixture weights collapse (batch entropy 2.05 → 0.27 nats; val perplexity
  1.5–2.3, 5.6 at j8) — in d = 196,608 the 8 IBW components (2–3 σ apart) differ by ~1e-4 nats/dim,
  so one slightly wider isotropic component is nearly forward-KL optimal.
* Stage B (val, 40 pairs) is at the IBW replicate floor at j0–j8: KL(q_IBW‖q_φ)/dim ≈ KL between two
  independent IBW fits of the same anchor (j8: 0.0086 vs 0.0105); DINOv2 B 0.02–0.12 vs replicate
  0.006–0.046. Terminal step weaker (B 0.25 vs 0.02; student pixel dispersion 0.76 of IBW's).
* End to end (n = 192, x_T 1,000,000+, preliminary FID): teacher 56.6; **untrained student
  (Tweedie mean + v_ref noise = the network's initialisation) 98.3**; ANC10 114.6; **trained pilot
  126.2**; RD10 451.7 (the 10-step SMLD predictor diverges, range ±1,000). Contrast (per-image pixel
  std): teacher 0.258, init 0.175, ANC10 0.174, trained 0.119 — training washes images out.
* Hybrids (trained student at the marked intervals, init elsewhere, identical draws; verified
  bit-exact at both ends): none 98.3, j9 only 101.4, j5–9 110.2, j0–4 111.3, j0–8 121.0, all 126.2
  → the degradation is distributed over the chain (≈ +13 high-noise half, +12 low-noise half).
* Per-interval comparison against the teacher (val 40 pairs, same metric): the Tweedie-mean kernel
  (untrained init) is CLOSER to the teacher's 200-step kernel than the IBW-25 targets — DINOv2 excess
  mean 0.219 vs 0.395 (trained student 0.356); pixel excess 0.0085 vs 0.0413 (j7: 0.015 vs 0.092;
  j8: 0.014 vs 0.136). Pixel dispersion vs teacher is the SAME for all three (0.99 at j0 → 0.69 at
  j7/j8): the diversity deficit belongs to every isotropic kernel; IBW's extra error is a BIAS of its
  means. Plausible mechanism: the IBW fixed point solves m = x + gap · E_q[s_θ(y, s)] (score at the
  END time, averaged over the fitted q), while the Tweedie mean is x + gap · s_θ(x, t) = E[y | x]
  under the model at the START time; the two agree only if q were the true conditional (tower
  property), and q lacks the content variance. The stiff (noise-like) directions, where this bias
  lives, converge within ~10 iterations, so neither the iteration count nor the initialisation
  removes it (P1: every init converges to the same fixed point).
* Rollout shift (student boundary states vs teacher anchors, n = 32): pixel dispersion ratio falls
  0.98 → 0.92 → 0.78 → 0.73 at boundaries 1 / 4 / 6 / 9 (teacher floor 0.98–1.08).
* Consequence: the main run (64 trajectories) and the endpoint-NLL ablation are run as planned; the
  ablation now isolates the IBW target bias (an NLL fit to teacher endpoints targets the teacher's
  conditional mean). The untrained initialisation is reported as a zero-training control.

## 2026-10-19 main run, first diagnostics (before the n = 1,024 FID)

* Main student (`student/main`, 64 train trajectories, 12,000 steps, 1.80 h on GPU0): val NLL
  1.4273 → 1.4084, flat from ≈ 4,500 steps; KL(q_IBW‖q_φ)/dim at the end: j5 0.0005, j7 0.003,
  j8 0.0079 (pilot 0.0086); weight entropy 0.23–0.29 nats on the last training batches (raw weights; collapse as in the pilot). At generation (EMA best checkpoint, student states): H(π) 0.27 at j0, 0.89–1.07 at j1–j9; draw perplexity 1.4–3.1.
* IBW chain (`ibwchain`, 64 images, frozen config online at all 10 transitions from the same x_T;
  8,001 teacher evaluations per image, 14,854 s on the shared GPU1 = 232 s/image; peak 33.6 GB):
  washed-out faces like the student's. n_eff of the online fits 1.0 at j0–j1 (components on top of
  each other), 4.7 at j2, 8 from j3; v/v_anc 1.00 (j0) … 1.04 (j7), 1.46 (j8), 2.45 (j9).
* FID at n = 64 (`fid_n64_ibwchain.json`, 500 paired bootstrap replicates; real-vs-real at n = 64
  is 90.8, so these values are dominated by small-n bias and only the paired differences mean
  anything): teacher 90.1, init 118.4, ANC10 136.5, IBW chain 141.7, pilot 142.5, main 152.4,
  RD10 459.3. Paired 95 % CIs: IBW chain − init [+16.6, +29.1]; main − init [+26.6, +41.8];
  main − IBW chain [+2.8, +18.3]; main − pilot [+3.9, +17.3]; IBW chain − pilot [−7.3, +6.8].
  Contrast: teacher 0.266, init 0.177, ANC10 0.177, IBW chain 0.141, pilot 0.122, main 0.121.
  → A perfect amortisation of the targets is no better than the untrained network; training
  longer on them (pilot → main) moves the sampler further from the teacher.
* Per-interval detail diagnostic (`gmmd/exp5/diag_detail.py`, val, 4 anchors × 10 intervals,
  32 draws per kernel, Tweedie-denoised end states; `figures/fig_detail_per_interval_val`):
  ratio to the teacher's 200-step kernel from the SAME anchor, contrast / Laplacian-RMS detail —
  IBW targets j1–j3 0.53–0.66 / 0.24–0.38, j4–j7 0.80–0.98 / 0.24–0.65, j8–j9 ≈ 1.0 / 1.01–1.15;
  untrained Tweedie Gaussian 0.89–1.00 / 0.87–1.04 at j1–j9; main student = IBW targets
  (0.56–0.98 / 0.21–0.65 at j1–j7). j0 (0.5 / 0.4 for every kernel) is the teacher-state denoising
  artifact. Interpretation (consistent with, not proof of, the mode-seeking hypothesis): the
  reverse-KL fit with a narrow isotropic family drives the means toward a MODE of the Bayes
  conditional; under a heavy-tailed image prior the mode shrinks small fine-scale coefficients
  (MAP vs MMSE), so each IBW transition removes detail and contrast that the teacher's interval —
  and the first-order Tweedie Gaussian — keep. The loss is concentrated at j1–j7 and compounds over
  the chain; j8–j9 cannot restore it.

## 2026-10-19 refinement (student-rollout anchors, same frozen IBW target construction)

* 32 rollouts of `student/main` (seeds 3,300,000+, split `refine`) → states at the starts of
  intervals 1–9 → 288 frozen IBW fits (prior-free, c = 0.5, 25 it, K = 8; 35 s per group of 4 on
  GPU0). All finite, none stopped early, none discarded. Compared with the training fits at the
  teacher's anchors: v/v_anc 0.99–1.00 at j1–j6 (training 1.00–1.03), median component
  separation at j5–j8 2.2 / 3.4 / 5.8 / 11.1 σ (training 3.5 / 7.5 / 12.8 / 22.7), occasional
  full merges at j3–j5 (2, 1, 1 / 32; none in training) → narrower conditionals at the smoother
  student states (interpretation).
* Fine-tune from `student/main/ckpt_best.pt`, original + refinement pairs (2 : 1 per interval at
  j1–j9), 4,000 steps, half learning rate, 2,205 s: val NLL (teacher anchors) 1.4084 → 1.4080, only
  j9 moves (−0.004); KL/dim flat.
* Rollout shift: pixel dispersion of boundary states vs teacher b5 0.76 → 0.99, b6 0.59 → 0.90,
  b9 0.51 → 0.78 (floor 1.03–1.08); pixel excess b9 0.038 → 0.021; DINOv2 still p = 0.002 at
  every boundary.
* FID n = 1,024 (`fid_n1024_refined.json`): refined 117.46 vs main 111.73, paired CI
  [+4.2, +8.2] — refinement HURTS end to end. Precision 0.649 → 0.679, recall 0, contrast
  0.125 → 0.124 (teacher 0.257), intensity SD over images 0.116 → 0.139 (teacher 0.120).
  → Rollout shift was a real but secondary problem; correcting it with more targets of the same
  (detail-losing) construction does not recover the images.

## 2026-10-19 post-hoc checks while the ablation runs

* Precision (`frozen_fp32`: the frozen configuration with IDENTICAL seeds, TF32 off; test trajs
  0–1 × 10 intervals; `gmmd/exp5/diag_precision.py`): component means move 0.0001–0.0008
  component SDs (fresh-seed replicate centroid: 0.0007–0.15 SDs), variance ratio 1.0000 ± 0.0002,
  paired stage-A excess fp32 − TF32: DINOv2 −0.0003 ± 0.0008 (mean 0.456), pixel −0.00002 ±
  0.00006 (mean 0.035). → TF32 is irrelevant to the target quality.
* Early test stage A/B/C (16–17 pairs, mostly trajs 0–1; preliminary, the full 80 pairs come after
  the test references): same pattern as the pilot on val — stage B small, stage C ≈ stage A for
  main, the untrained init closer to the teacher than the IBW targets at j2–j8.
* Final images: the Laplacian energy of every student's output is nearly constant over images
  (init 0.062 ± 0.002, main 0.060 ± 0.005), so I suspected residual noise from the terminal
  targets (IBW j9 fit SD 0.0135 per pixel > σ_min; T(z) bank spread 0.0083 train / 0.0101 refine).
  REFUTED as a major factor (`gmmd/exp5/diag_terminal.py`, 4 train anchors × 8 rollouts,
  `figures/diag/terminal_spread.json`): from the same j9 anchor the TEACHER's 200-step terminal
  kernel leaves a per-pixel spread of 0.0107 around its own mean (Laplacian 0.039) — more than the
  IBW bank 0.0088, init 0.0085, main 0.0091, refined 0.0090; kernel means differ from the teacher's
  by 0.0054–0.0071 RMS. Noise-free last step (diagnostic `student:<tag>:best:meanlast`, same x_T,
  components and states; `fid_n1024_meanlast.json`): main 111.73 → 110.16 (paired [+1.3, +1.9]),
  refined 117.46 → 115.80 ([+1.3, +2.0]), init 77.61 → 75.15 ([+2.1, +2.6]). The final draw costs
  ≈ 2 FID points; the 85-point gap to the teacher is elsewhere (mid-noise detail loss, §3).
  Refined images: Laplacian 0.086 ± 0.045 (teacher 0.090 ± 0.023) at contrast 0.124, 0.073 with
  the noise-free last step → their extra high-frequency energy is mostly not the last draw; grain
  is visible in `figures/diag/fig_samples_same_xT_refined`.

## 2026-10-20 ablation (same network, NLL on E = 4 fp32 teacher endpoints per training pair)

* Endpoints: 640 pairs × 4 rollouts, fp32 (done 04:05 UTC). Training: val NLL of held-out teacher
  endpoints (val reference sets) 1.4110 → **1.4057 at step 1,750**, back to 1.4104 at 12,000 while
  the training loss falls 1.419 → 1.411 → overfits after ~2,000 steps (2,560 target images for
  66 M parameters); `ckpt_best` = step 1,750. Its v_ref,j / v_anc,j = 1.02 … 1.26 (j0–j7), 1.53
  (j8), 1.13 (j9) vs main's (IBW variances) 1.00 … 1.11, 1.60, 0.83.
* FID n = 1,024 (`fid_n1024_students.json`): **ablation 74.34 ± 0.85** (P 0.660, R 0.017, contrast
  0.176, Laplacian 0.076, mean intensity 0.362 — darker than the teacher's 0.418; 3.1 % of raw
  values clipped). Paired: ablation − main [−38.9, −35.0]; − init [−4.2, −1.0]; − ANC10
  [−21.9, −16.9]; teacher − ablation [−50.5, −46.6].
* Component usage (sampler draws): H(π) 1.69–1.91 nats, draw perplexity 5.6–6.8 at every interval
  (main 1.4–3.1; init 8.0 = uniform).
* Held-out likelihood of the teacher's transitions (`gmmd/exp5/diag_nll.py`, test, PRELIMINARY on
  the 61 pairs with references at 06:15 UTC; rerun on all 80): per-dim NLL in v_anc units —
  j0–j2 all equal (±0.001); j3–j7 ablation < main ≈ refined < IBW ≈ init (e.g. j7 1.5255 / 1.5317 /
  1.5363 / 1.5367); j8 ablation 1.621 < init 1.629 < main 1.655 < IBW 1.670; j9 ablation 1.419 <
  init 1.468 < main 1.586. Decomposition per coordinate, relative to the teacher's variance τ:
  total variance IBW / init / main 0.98 → 0.89 (j0 → j7), ablation 1.00–1.02; squared mean bias
  IBW 0.000–0.013 at j0–j7 (SMALLER than init's 0.000–0.015 at j2–j7), but **0.10 at j8** (init
  0.016, ablation 0.0015) and main's terminal kernel **0.31 at j9** (init 0.05, ablation 0.005).
  → In pixel space the IBW means are close to the teacher's mean at j1–j7; the detail loss there
  (§3 of REPORT) is invisible to L2 at these noise levels (small coefficients), while the IBW mean
  bias proper sits at j8 and in the terminal pushforward. Earlier pooled statement "IBW's extra
  error is a bias of its means" (pilot, pixel excess averaged over intervals) is dominated by j8–j9.
