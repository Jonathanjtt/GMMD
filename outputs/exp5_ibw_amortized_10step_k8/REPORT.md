# exp5 — Amortised IBW–GMMD: ten 8-component mixture draws in place of the 2,000-step teacher

**Question.** Can our existing IBW conditional-mixture fitting be amortised into a neural network,
so that ten 8-component Gaussian-mixture draws replace the complete diffusion sampling trajectory
with useful fidelity and lower inference cost?

**Short answer.** _(filled in at the end of the run from the numbers below)_

---

## 1. Method

```
offline A   teacher trajectories ──► anchors x at the 10 interval starts (independent seeds per split)
            per anchor and interval: IBW fit of the VE Bayes conditional, K = 8  ──► cached q_IBW(· | x, j)
offline B   conditional mixture network q_φ(· | x, j), trained on fresh draws z ~ q_IBW (forward KL)
inference   x_T ~ N(0, σ_max² I) ──► for j = 0..9: one network evaluation, one component, fresh noise ──► image
```

What the existing IBW implementation is, how it obtains its gradient-flow information, which
parameters it updates and which covariances it supports is recorded in `DESIGN.md` §1 (inspected
before any large run). In one line: IBW descends KL(q ‖ p) for the continuous-time VE Bayes
conditional `p(y | x) ∝ p_s(y) N(x; y, (σ_t² − σ_s²) I)` using only the score
`s_θ(y, s) + (x − y)/(σ_t² − σ_s²)`, over uniform-weight (fixed 1/K) isotropic mixtures, means by
gradient descent and variances by Eq. (12). This is an approximation of the teacher's 200-step
SMLD kernel, not that kernel; stage A below measures the gap.

**Teacher (frozen).** CelebA-HQ 256 NCSN++, VE SDE, EMA weights; predictor-only reverse diffusion,
2,000-point grid, prior `x_T = 348 z`, noise removal at the end — exp4's original pipeline,
unchanged. Teacher reference samples are fp32 with TF32 off; IBW fitting and the student use TF32
(user decision).

**Partition.** 2,000 steps → 10 contiguous intervals of 200 steps (`partition.md`); interval j
lowers σ by ≈ 2.85×. The terminal interval runs the stochastic steps to the σ_min = 0.01 state and
then the teacher's own deterministic noise removal `T(x) = x + σ_min² s_θ(x, ε)`: IBW fits the
σ_min state's conditional, the student's 10th kernel is trained on `T(z)`, `z ~ q_IBW` (computed
offline), so its draw is already a final image and inference makes no teacher call. The common
post-processing of every method is upstream's quantisation `clip(255 x).astype(uint8)`.

**IBW step size and iterations.** `γ_j = c · v_anc,j` with `v_anc = σ_s² gap / σ_t²` (the
Gaussian-data posterior variance), the only per-interval change; Eq. (12) and the mean update are
untouched. c and the iteration budget come from the P0 check; the initialisation and iteration
count from the P1 validation selection (pre-registered rule, `DESIGN.md` §6).

**Student.** The teacher's NCSN++ initialised from its EMA weights with the 7 output-skip
convolutions widened to 8 × 3 channels (one score-like field per component) and a global head
(8 mixture logits + 8 log-variance offsets). `μ_k = x + g_j h_k`, isotropic `v_k`, weights global
over the image. At initialisation every component is the teacher's first-order Tweedie mean of the
interval's end state.

## 2. Offline IBW targets (stage A on validation anchors)

All numbers in this section are on the 4 validation trajectories × 10 intervals = 40 (anchor,
interval) pairs, against two independent 32-sample sets of 200-step teacher rollouts per pair
(fp32). Excess = MMD²(IBW, A) − MMD²(B, A); DINOv2 features of Tweedie-denoised states, plus 8×
pooled pixels. Tables: `fitsummary_val_*.md`; selection: `selection.json`.

**P0 — step size (exploratory, before P1).** Prior-free K = 8 fits at intervals 1/4/7, c ∈ {0.2,
0.5, 1.0}, 400 iterations: every fit reaches a gradient plateau by 50–100 iterations, the converged
fidelity does not depend on c (mean excess 0.58 / 0.57 / 0.59), the stationary jitter grows with c.
Chosen: c = 0.5, budget 300 (`figures/fig_p0_ibw_step_size`).

**P1 — initialisation (user decision: teacher-sample-free, components far apart, initial mean and
covariance tuned on validation).** Five variants × 40 pairs × 300 iterations, scored at 25 / 50 /
100 / 200 / 300 iterations (`figures/fig_p1_stage_a_vs_iterations`):

| variant | init means | init var | best snapshot | mean DINOv2 excess (best) | at 300 it |
|---|---|---|---|---|---|
| prior_free | x + √gap · z | gap | 25 | **0.391** | 0.625 |
| principal_r3_vgap | Tweedie mean ± 3√λᵢ eᵢ | gap | 25 | 0.550 | 0.619 |
| principal_r3 | Tweedie mean ± 3√λᵢ eᵢ | v_anc | 25 | 0.575 | 0.619 |
| principal_r1 | Tweedie mean ± √λᵢ eᵢ | v_anc | 25 | 0.578 | 0.617 |
| random_far | Tweedie mean + √(d v_anc) uᵢ (≈ 313 σ apart) | v_anc | 25 | 0.579 | 0.617 |

The pre-registered rule froze **prior_free at 25 iterations** (paired bootstrap of every other
variant − prior_free: +0.16 … +0.19, all 95 % CIs exclude 0). Two facts matter for everything
that follows:

* *All initialisations reach the same fixed point* (300 it: 0.617–0.625, n_eff ≈ 6.6, v / v_anc
  from 1.00 at j0 to 1.23 at j9; random_far's 313 σ separations collapse to 0.5–4 σ). Converged by
  the pre-registered drift test: 23/40 (prior_free) — all of j0–j4, 3/4 of j5, none of j6–j9 where
  the Monte-Carlo noise floor of B = 4 gradient samples exceeds the tolerance; a secondary
  gradient-plateau test classifies 36/40 as stationary.
* *The selected snapshot is a transient, not a converged fit.* The prior-free start puts every
  mean at x + √gap z with variance gap; the mean update contracts stiff (noise-like) directions at
  rate ≈ c per iteration but soft content directions only at ≈ c · v_anc / gap ≈ 0.06, so after 25
  iterations ~20 % of the random content offsets survive. A post-hoc fine grid on the same pairs
  with fresh seeds (`figures/fig_posthoc_ibw_iterations`; sensitivity only, `selection.json`
  unchanged) shows a broad optimum: 20 / 25 / 30 iterations give 0.354 / 0.393 / 0.406 (25 and 30
  not separable from 20), 5–15 and 40–50 are significantly worse; pixel excess falls monotonically
  (0.385 at 5 → 0.042 at 25 → 0.020 at 50). Fresh-seed fits reproduce P1 at 25 it (0.393 vs 0.391,
  per-pair correlation 0.995).

**Absolute quality of the targets.** Even the selected targets are far from the teacher's kernel
at mid and low noise: per-interval DINOv2 excess 0.13–0.30 at j3–j7, 0.31–0.45 at j8–j9 (≈ 0.7 at
j0–j2 is dominated by an artifact: the teacher's own states denoise to colour-noisy images, so
DINOv2-on-Tweedie cannot be read as fidelity there; pixel excess ≈ 0 at j0–j3). Every isotropic
kernel is under-dispersed in content directions (pixel dispersion vs teacher 0.92 at j4 → 0.62 at
j9; `figures/diag/samples_val000_j{4,7,9}_*.png`: at j4 the teacher's interval produces different
faces, every IBW snapshot near-identical blurry ones).

## 3. Pilot (P2): amortisation works, the targets do not

Pilot student: 8 training trajectories (80 pairs), 3,000 steps; evaluated on the 40 val pairs and
192 images (x_T seeds 1,000,000+, the teacher's exp4 seeds; FID at n = 192 is preliminary and
biased upward — the teacher itself scores 56.6).

* **Amortisation (stage B) is at the IBW floor for j0–j8.** KL(q_IBW ‖ q_φ) per dimension equals the
  KL between two independent IBW fits of the same anchor (e.g. j8: 0.0086 vs 0.0105); sample
  DINOv2 excess 0.02–0.12 vs 0.006–0.046 for the IBW replicate. The terminal step is weaker
  (0.25 vs 0.02). The learned weights collapse to 1.5–2.3 effective components (5.6 at j8): in
  d = 196,608 the 8 IBW components differ by ~10⁻⁴ nats per dimension, so a single slightly wider
  isotropic component is nearly forward-KL optimal.
* **End to end, training on the IBW targets makes the sampler worse than its own
  initialisation.** FID (n = 192): untrained student 98.3 (its initialisation is the teacher's
  Tweedie mean of each interval plus isotropic noise), 10-step ancestral 114.6, **trained pilot
  126.2**, 10-step SMLD predictor 451.7 (diverges). Per-image contrast falls from 0.175 (init) to
  0.119 (teacher 0.258). Hybrid samplers (trained student at some intervals, init elsewhere,
  identical draws): the damage is spread over the chain (high-noise half +13 FID, low-noise half
  +12, terminal step +3).
* **Why: the IBW targets are worse conditional approximations than the trivial Tweedie Gaussian.**
  Scoring the untrained init with the same stage-A metric: mean DINOv2 excess 0.219 (Tweedie
  Gaussian) vs 0.395 (IBW-25 targets) vs 0.356 (trained student); pixel excess 0.0085 vs 0.0413
  (j8: 0.014 vs 0.136). Their pixel dispersions relative to the teacher are identical, so IBW's
  extra error is a **bias of its means**. Plausible mechanism: the IBW fixed point satisfies
  m = x + gap · E_q[s_θ(y, s)] (the score at the interval's END time averaged over the fitted q),
  while the Tweedie mean x + gap · s_θ(x, t) is E[y | x] under the model; they coincide only if q
  were the true conditional, and the isotropic q lacks its content variance. The bias lives in the
  stiff directions that converge within ~10 iterations, so neither initialisation nor iteration
  count (P1, fine grid) removes it.

_(sections 4–8 — main run on held-out test anchors, 1,024-image FID, shift/refinement, ablation,
cost — are filled from the result files at the end of the run)_


---

## Reproduction

Everything runs from the repository root on branch `exp5-ibw-amortized`; `M="python -m
gmmd.experiments.ibw_amortized"` (config `exp5_ibw_amortized_10step_k8`, outputs under
`outputs/exp5_ibw_amortized_10step_k8/`). Every stage is resumable (it skips finished units) and
claim-safe across GPUs (`locks/`); seeds are derived from `(namespace, split, trajectory, interval,
variant)` so reruns reproduce the same fits and samples.

| step | command | notes |
|---|---|---|
| partition, split manifests | `$M partition` | `partition.json`, `manifests/` |
| teacher anchors (fp32) | `$M anchors --split {train,val,test}` | states at the 10 interval starts |
| teacher references (fp32) | `$M refs --split {val,test}` | 2 × 32 rollouts per (anchor, interval) |
| P0 / P1 IBW fits + stage A | `bash run/exp5_pilot_p0.sh`, `bash run/exp5_pilot_p1.sh` (+ `run/exp5_pilot_p1_finish.sh`) | ends with `$M select` → `selection.json` |
| freeze + pilot | `bash run/exp5_pilot_p2.sh` | `$M freeze --split val`, pilot fits, pilot student, val stage A/B/C, 192 images |
| training targets | `$M fit --split train --variant frozen` | 64 × 10 fits, 25 iterations, terminal banks |
| held-out targets + replicate | `$M fit --split test --variant frozen`, `... --variant frozen_r1` | |
| main student | `$M train --tag main` | resumes from `student/main/ckpt_last.pt` |
| standalone sampling | `python -m gmmd.exp5.sampler --ckpt outputs/exp5_ibw_amortized_10step_k8/student/main/ckpt_best.pt --n 64 --out samples.npz` | 10 network evaluations, 0 teacher calls |
| images for FID | `$M generate --method student:main --n 1024` (also `coarse:<kind>:<steps>`, `ibwchain`, `hybrid:<a>:<b>:<mask>`) | x_T seeds 1,000,000+ (= exp4's teacher images) |
| FID / P-R | `$M fid --tag <tag> --n 1024 --methods ...` | paired bootstrap vs the teacher |
| stage A/B/C on test | `$M evalstudent --tag main --split test` | uses `refs/test`, `fits/test/frozen{,_r1}` |
| shift, refinement | `$M shift --method student:main`; `run/exp5_main_part2.sh` (refinement block) | |
| ablation | `$M endpoints --split train --n-endpoints 4`; `$M train --tag ablation_endpoints --data endpoints` | |
| timing | `bash run/exp5_timing.sh` | uncached, synchronised, fp32 and TF32 |
| tables, costs, figures | `$M tables`, `$M costs --student main refined`, `python -m gmmd.exp5.figures` | |

The full main run as executed: `run/exp5_main.sh` (both chains; GPU0's chain was handed over to
`run/exp5_main_part2.sh` after the main student's shift — see RUNLOG).
