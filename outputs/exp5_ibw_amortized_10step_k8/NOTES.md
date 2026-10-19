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
