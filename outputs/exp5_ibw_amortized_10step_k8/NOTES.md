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
