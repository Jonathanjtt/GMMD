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
  count (P1, fine grid) removes it. (The held-out decomposition in §4 localises it: in pixel space
  the IBW mean bias exceeds the Tweedie mean's only at j8 and in the terminal pushforward; at j1–j7
  the IBW means are as close to the teacher's mean as the Tweedie mean or closer, and what they
  lose there is fine detail — next bullets.)
* **A perfect amortisation of these targets would not help (IBW chain).** Running the frozen IBW
  configuration online at every one of the ten transitions — at the chain's own current state, so
  there is neither network error nor rollout shift; uniform weights, one draw, the teacher's noise
  removal at the end (`gmmd/exp5/oracle.py`) — costs 8,001 teacher evaluations per image (4× the
  teacher's 2,000; 232 s per image on the shared GPU 1) and gives the same washed-out faces as the
  student (`figures/fig_samples_same_xT_prelim`): contrast 0.141 vs the teacher's 0.266 on the same
  64 x_T. FID at n = 64 (diagnostic only; heavily biased — CIs in `fid_n64_ibwchain.json`):
  teacher 90.1, untrained init 118.4, 10-step ancestral 136.5, **IBW chain 141.7**, pilot student
  142.5, main student 152.4, 10-step SMLD 459.3. End-to-end quality is therefore bounded by the
  targets (stage A), not by the network (stage B).
* **Where the targets lose the images: fine detail at mid noise.** Per interval, from the same
  anchor, the Tweedie-denoised end states of each kernel against the teacher's 200 steps
  (`gmmd/exp5/diag_detail.py`, `figures/fig_detail_per_interval_val`; ratio kernel / teacher,
  contrast / Laplacian-RMS detail): IBW targets 0.53–0.66 / 0.24–0.38 at j1–j3 and 0.80–0.98 /
  0.24–0.65 at j4–j7, ≈ 1 at j8–j9; the untrained Tweedie Gaussian stays at 0.87–1.04 throughout
  j1–j9; the trained student reproduces the IBW targets. This is what a mode-seeking fit predicts:
  reverse KL with a narrow isotropic family drives the means toward a mode of the Bayes conditional,
  and under a heavy-tailed image prior the mode shrinks fine-scale coefficients (MAP vs posterior
  mean). It is consistent with — not a proof of — that mechanism.

## 4. Main run: targets, training, held-out amortisation

**Training targets.** 64 independent teacher trajectories × 10 intervals = 640 frozen IBW fits
(prior-free init, c = 0.5, 25 iterations, K = 8, B = 4 gradient samples; 5.14 × 10⁵ teacher
evaluations, 3.1 GPU-h with TF32), plus 8 held-out test trajectories (80 fits) and an independent
replicate of each test fit (`frozen_r1`, fresh init and noise seeds). Every fit is finite with
positive variances and none stopped early. None is converged by the pre-registered drift test —
by design: the selected 25-iteration snapshot is an early-stopped transient, and the converged
300-iteration fits scored worse on validation (§2). Nothing was discarded. Per interval (train;
test agrees within a few %): v / v_anc = 1.00–1.03 at j0–j6, 1.10 at j7, 1.60 at j8, 2.08 at j9;
the components start ≈ 313 σ apart and end 0.55 σ apart at j0–j1 (all 8 on top of each other,
n_eff = 1.0), 1.4 σ at j2 (n_eff 5.0) and 1.9–37 σ at j3–j9 (n_eff 8.0); the mean-gradient norm
has fallen to 0.1 % (j0) … 24 % (j9) of its initial value. Tables: `fitsummary_*`, records under
`fits/<split>/frozen/`.

**Student training** (`student/main`, `figures/fig_training_main`): 12,000 steps (1.8 GPU-h); batches
draw an interval uniformly, then a pair, then fresh samples from the cached mixture (terminal
interval: rows of its cached 32-sample pushforward bank, since `T` needs the teacher). Validation NLL 1.4273 → 1.4084 nats/dim, flat from
≈ 4,500 steps (the best checkpoint is step 4,750); KL(q_IBW ‖ q_φ) per dimension at the end:
≤ 0.0005 at j0–j5, 0.0013 (j6), 0.0033 (j7), 0.0079 (j8). **Component usage** (the sampler's own
draws, 1,024 images, best checkpoint): the per-state weight entropy is 0.27 nats at j0 and
0.89–1.07 at j1–j9 (uniform: ln 8 = 2.08), and over images two or three components take almost
every draw (perplexity of the drawn components 1.4 at j0, 2.4–3.1 at j1–j9; components 3 and 6
at j3–j7, component 7 at j8–j9). The weights are not uniform like the IBW targets' because in
d = 196,608 the eight target components differ by ~10⁻⁴ nats per dimension and a few slightly
wider isotropic components are nearly forward-KL optimal (§3). Every variance is positive by construction
(`log v = log v_ref + 6 tanh(·/6)`); every generated image is finite (raw range −0.70 … 1.48,
≤ 3.1 % of values clipped by the common uint8 quantisation).

_(Held-out test evaluation — stage A/B/C with the replicate floor for main, refined, ablation and
the untrained init — fills this section when the test references finish.)_

## 5. End to end: 1,024 images per method

Same 1,024 x_T as the teacher (seeds 1,000,000+), identical preprocessing, FID / precision /
recall against all 30,000 CelebA-HQ 256 images, paired bootstrap over images; real-vs-real FID at
n = 1,024 is 15.1, so every value carries a common small-n bias and only paired differences are
interpreted.

| method | network evals / image | teacher evals / image | FID ↓ | precision | recall | contrast |
|---|---|---|---|---|---|---|
| teacher (2,000-step SMLD) | 2,000 | 2,000 | 22.55 ± 0.71 | 0.769 | 0.464 | 0.257 |
| untrained init (Tweedie Gaussian) | 10 | 0 | 77.61 ± 0.80 | 0.577 | 0.001 | 0.176 |
| ancestral, 10 steps | 10 | 10 | 94.38 ± 0.99 | 0.522 | 0.000 | 0.174 |
| **IBW–GMMD student (main)** | 10 | 0 | **111.73 ± 0.81** | 0.649 | 0.000 | 0.125 |
| SMLD predictor, 10 steps | 10 | 10 | 442.16 ± 0.62 | 0.000 | 0.000 | 0.500 |

Paired 95 % intervals: main − ancestral-10 [+15.0, +19.9]; main − untrained init [+32.5, +36.2];
untrained init − ancestral-10 [−19.3, −14.2]. The amortised IBW–GMMD sampler is significantly
worse than the cheapest supported 10-evaluation teacher sampler and than its own initialisation.
(`fid_n1024_first.json`, `fid_n1024_init.json`; sample grids `figures/fig_samples_same_xT_prelim`;
intermediate states `figures/fig_trajectory_student_main`.)

**The last draw is not the problem** (post-hoc diagnostic). From the same terminal-interval
anchor, the teacher's own 200-step terminal kernel leaves a per-pixel spread of 0.0107 around its
mean — more than the IBW terminal targets (0.0088) and the students' last kernels (0.0085–0.0091)
(`gmmd/exp5/diag_terminal.py`, 4 anchors × 8 rollouts, `figures/diag/terminal_spread.json`).
Returning the drawn component's mean at the last step instead of a draw (same x_T, components and
states; a diagnostic, not the method) lowers FID by 1.6 for main (110.16, paired [+1.3, +1.9]), 1.7
for refined and 2.5 for the untrained init (`fid_n1024_meanlast.json`): the final noise accounts
for about 2 of the 89 FID points between the student and the teacher.

Refined student (§6): FID 117.46 ± 1.16, P 0.679, R 0.000, contrast 0.124; refined − main
[+4.2, +8.2] (`fid_n1024_refined.json`). _(Endpoint-NLL ablation, ancestral 20 / 50 steps:
pending — one final FID file with every method.)_

## 6. Rollout shift and refinement

Boundary states of 32 student rollouts against the teacher's anchors at the same σ (DINOv2 on
Tweedie-denoised states and 8× pooled pixels; teacher-vs-teacher floor): the main student's pixel
dispersion falls to 0.98 (b1), 0.90 (b4), 0.59 (b6) and 0.51 (b7–b9) of the teacher's (floor
0.99–1.08; pilot: 0.73 at b9), with pixel excess MMD² 0.038 at b7–b9 (pilot 0.017). The degradation
is substantial, so the refinement stage runs as specified: 32 student-rollout trajectories
(`refine` split, seeds 3,300,000+) → frozen IBW fits at their states, intervals 1–9 (288 fits, the
same target construction) → fine-tuning from the main checkpoint on the original and the
refinement targets together (2 : 1 per interval at j1–j9, j0 unchanged; 4,000 steps, half learning
rate). Validation, test and the 1,024 evaluation seeds are untouched.

**Refinement targets.** 288 / 288 fits finite with positive variances, none stopped early, none
discarded. At the student's own states the fits are narrower and less spread than at the
teacher's anchors: v / v_anc 0.99–1.00 at j1–j6 (training fits 1.00–1.03), median component
separation 2.2 / 3.4 / 5.8 / 11.1 σ at j5–j8 (training fits 3.5 / 7.5 / 12.8 / 22.7 σ), and a few
anchors with all eight components merged at j3–j5 (2, 1, 1 of 32; never in the training fits),
consistent with the smoother student states having narrower Bayes conditionals. Fine-tuning moved the teacher-anchor
validation NLL only from 1.4084 to 1.4080 nats/dim, all of it at the terminal interval
(`figures/fig_training_refined`): the refinement targets teach the network about the states it
visits without costing fidelity at the teacher's anchors.

**Pre / post refinement** (`eval/shift_student_{main,refined}.json`; n = 32 rollouts vs 32 + 32
teacher anchors; dispersion = mean pairwise distance within a set, relative to the teacher's;
floor = the second teacher set):

| boundary | σ | pixel dispersion main → refined (floor) | pixel excess MMD² main → refined | DINOv2 excess main → refined |
|---|---|---|---|---|
| b1 | 122 | 0.98 → 0.99 (0.99) | 0.0006 → 0.0007 | 0.64 → 0.16 |
| b2 | 43 | 0.97 → 0.99 (1.00) | −0.0006 → −0.0007 | 0.58 → 0.46 |
| b3 | 15 | 0.94 → 0.98 (1.00) | 0.0005 → 0.0007 | 0.17 → 0.20 |
| b4 | 5.3 | 0.90 → 1.00 (1.01) | 0.0023 → 0.0013 | 0.07 → 0.06 |
| b5 | 1.9 | 0.76 → 0.99 (1.03) | 0.0103 → 0.0034 | 0.10 → 0.05 |
| b6 | 0.65 | 0.59 → 0.90 (1.07) | 0.0290 → 0.0136 | 0.11 → 0.08 |
| b7 | 0.23 | 0.52 → 0.82 (1.08) | 0.0367 → 0.0193 | 0.13 → 0.12 |
| b8 | 0.081 | 0.51 → 0.80 (1.08) | 0.0383 → 0.0205 | 0.21 → 0.16 |
| b9 | 0.028 | 0.51 → 0.78 (1.08) | 0.0384 → 0.0210 | 0.22 → 0.24 |

Refinement halves the low-noise pixel shift and restores the dispersion through b5, but the
boundary states remain under-dispersed from b6 on and DINOv2-distinguishable from the teacher's
at every boundary (p = 0.002, the permutation minimum, for both students; the pixel tests at b5–b9
move from p ≤ 0.05 to p = 0.06–0.32 — not significant at n = 32, which is not equality).

**End to end, refinement makes the images worse** (`fid_n1024_refined.json`, same 1,024 x_T,
paired bootstrap): FID 117.46 ± 1.16 vs 111.73 ± 0.81 for main, refined − main [+4.2, +8.2];
refined − ancestral-10 [+20.8, +26.6]; refined − untrained init [+37.8, +42.9]. Precision rises
(0.649 → 0.679), recall stays 0, per-image contrast is unchanged (0.124 vs 0.125; teacher 0.257),
the spread of mean intensity over images grows past the teacher's (SD 0.116 → 0.139; teacher
0.120), and more raw values leave [0, 1] before the common uint8 quantisation (3.1 % of values
clipped vs 1.0 % for main, raw range −3.9 … 2.4 vs −0.7 … 1.5; every image finite). Refinement corrects the coarse statistics of where the rollouts go — the boundary
dispersion above — but not what each IBW target does to fine detail (§3), and FID does not reward
the trade (`figures/fig_trajectory_student_refined`: the refined rollouts are still smooth,
low-contrast faces; `figures/diag/fig_samples_same_xT_refined`: on the same x_T the refined
images are as washed out as main's and carry visible high-frequency colour grain).

## 7. Ablation: the same network trained on teacher endpoints

**Setup.** Same architecture, initialisation (teacher EMA), optimiser, schedule and 12,000-step
budget as main; the targets are E = 4 fp32 teacher rollouts of each interval from each of the 640
training anchors (`endpoints/train/`), i.e. 800 teacher evaluations per (anchor, interval) — the
budget of one IBW fit (25 iterations × 8 components × 4 gradient samples). Loss: the same
standardised NLL, now of the cached endpoints (rows drawn with replacement); `v_ref,j` is the median
spread of the endpoints. Model selection uses the validation anchors' teacher reference sets, so
no IBW information reaches the ablation. It is an ablation that bounds what the target
construction costs, not the method.

**Training** (`student/ablation_endpoints`, `figures/fig_training_ablation_endpoints`): validation
NLL of held-out teacher endpoints 1.4110 → 1.4057 nats/dim at step 1,750, then back up to 1.4104
by 12,000 while the training loss keeps falling (1.419 → 1.411): four endpoints per pair (2,560
target images for 66 M parameters) overfit after ~2,000 steps, and `ckpt_best` is step 1,750.
(These NLLs are not comparable with main's 1.4084, which scores IBW draws in different
standardised units.) Unlike main, the ablation uses its components: weight entropy 1.69–1.91
nats (uniform 2.08), drawn-component perplexity 5.6–6.8 at every interval (main 1.4–3.1), and a
weight-weighted separation of 13–20 component SDs between component means (test anchors,
`figures/diag/nll_teacher_endpoints_test.json`). Its kernels' total per-coordinate variance
matches the teacher's 200-step kernel (1.00–1.02 of it at j0–j7), where the IBW targets, main and
the untrained init all have 0.89–0.98 — maximum likelihood matches second moments, the
mode-seeking IBW fit does not.

**End to end** (`fid_n1024_students.json`, the same 1,024 x_T, paired bootstrap):

| method | FID ↓ | precision | recall | contrast | Laplacian RMS | mean intensity | clipped |
|---|---|---|---|---|---|---|---|
| teacher | 22.55 ± 0.71 | 0.769 | 0.464 | 0.257 | 0.089 | 0.418 | — |
| **ablation (endpoint NLL)** | **74.34 ± 0.85** | 0.660 | 0.017 | 0.176 | 0.076 | 0.362 | 3.1 % |
| untrained init | 77.61 ± 0.82 | 0.577 | 0.001 | 0.176 | 0.062 | 0.424 | 0.01 % |
| ancestral, 10 steps | 94.38 ± 0.99 | 0.522 | 0.000 | 0.174 | 0.016 | 0.407 | 0 % |
| IBW–GMMD main | 111.73 ± 0.81 | 0.649 | 0.000 | 0.125 | 0.060 | 0.397 | 1.0 % |

Paired 95 % intervals: ablation − main [−38.9, −35.0]; ablation − untrained init [−4.2, −1.0];
ablation − ancestral-10 [−21.9, −16.9]; teacher − ablation [−50.5, −46.6].

**Reading.** With the teacher's own transitions as targets, the same network and the same
ten-draw sampler are 37 FID better than with the IBW targets — the targets, not the network or
the sampler, limit the main method (as stage B at the replicate floor and the online IBW chain
already indicated, §3). But the ablation is only 3.3 FID better than its untrained initialisation
and 52 behind the teacher: ten isotropic K = 8 transitions fitted by likelihood to four endpoints
per anchor reproduce the teacher kernels' second moments, not their structure. Its images keep
more contrast and fine detail than main's but are darker than the teacher's (mean intensity 0.362
vs 0.418), clip more (3.1 % of raw values outside [0, 1], range −0.56 … 1.05), and some rollouts
drift into dark, posterised images (`figures/fig_trajectory_student_ablation_endpoints`, row 1).
Not tried, for budget: more endpoints per anchor (E = 4 overfits), more anchors, or tuning — the
ablation ran with main's hyperparameters.

**Where along the chain the IBW targets cost FID** (post-hoc diagnostic; `run/exp5_hybrid_ablation.sh`,
`fid_n1024_hybrid_ablation.json`). Hybrid samplers run the ablation's network at one block of
intervals and main's elsewhere (swap-in), or main's at one block and the ablation's elsewhere
(swap-out), with the same prior, x_T and per-image draw generators as the standalone sampler, so
the two pure chains reproduce main and the ablation exactly and every difference comes from
which network makes which transition. 1,024 images each, paired bootstrap:

| block | σ | swap-in: main + ablation block — FID (gain vs main) | contrast | swap-out: ablation + main block — FID (loss vs ablation) | contrast |
|---|---|---|---|---|---|
| high noise j0–j2 | 348 → 15 | 108.50 (+3.2 [+1.3, +4.9]) | 0.119 | 80.23 (+5.9 [+3.8, +7.0]) | 0.176 |
| mid j3–j5 | 15 → 0.65 | **87.64 (+24.1 [+22.6, +25.1])** | 0.168 | **91.70 (+17.4 [+15.4, +18.5])** | 0.126 |
| low j6–j8 | 0.65 → 0.028 | 99.17 (+12.6 [+11.9, +13.7]) | 0.131 | 80.87 (+6.5 [+5.8, +7.6]) | 0.167 |
| terminal j9 | 0.028 → image | 107.46 (+4.3 [+4.0, +4.6]) | 0.125 | 77.78 (+3.4 [+3.1, +3.6]) | 0.177 |
| (pure chains) | | main 111.73 | 0.125 | ablation 74.34 | 0.176 |

Every block matters (all eight intervals exclude 0), but the mid-noise block j3–j5 carries the
largest share in both directions: the ablation's j3–j5 alone recovers 24 of main's 37 missing FID
points and nearly all of the contrast (0.125 → 0.168), and main's j3–j5 alone costs the ablation
17 points and all of its contrast (0.126) — the washed-out look is made between σ = 15 and 0.65,
where §3 found the IBW targets keeping 0.24–0.65 of the teacher's fine detail. Next is the
low-noise block (13 / 6.5), then the terminal step (4.3 / 3.4: main's last kernel has the largest
mean bias of any kernel, §4, but it costs only a few FID points, as the noise-free-last-step check
in §5 also found) and the high-noise block (3.2 / 5.9). The high-noise block sets global
brightness: with the ablation's j0–j2 the images are dark (mean intensity 0.351 in main + ablation
j0–j2, 0.362 for the pure ablation) and with main's they are not (0.412 in ablation + main j0–j2;
main 0.397, teacher 0.418), so the ablation's dark, posterised rollouts start at high noise. The
blocks are not additive — the four swap-in gains sum to 44, the four swap-out losses to 33, around
the 37-point total — so a single number per block is only approximate (mean of the two: 4.5 /
20.8 / 9.6 / 3.9).

## 8. Cost

_(Pending timing.)_ Offline so far (`costs.md`): training data 5.5 GPU-h (teacher anchor
trajectories 2.4, IBW fits 3.1), student training 1.8 GPU-h; one-off P0/P1 selection 14.0 GPU-h;
evaluation-only work 3.6 GPU-h and counting. The standalone sampler's record for every generated
chunk: 10 network evaluations and 10 mixture draws per image, 0 teacher score evaluations, 0
teachers constructed, 0 IBW iterations, inference mode, no gradients, 0 optimizer steps.

## 9. Limitations

* **The target family is fixed by the existing IBW implementation**: uniform-weight isotropic
  mixtures, fitted by reverse KL. No richer covariance was tried, because the implementation does
  not support one and the task forbids silently replacing it; the student uses the same isotropic
  family per component. The detail loss of §3 is consistent with a structural bias of exactly this
  combination (mode-seeking reverse KL, one variance per component in d = 196,608), which the knobs
  the implementation exposes — step size, initialisation, initial covariance, iteration count —
  did not remove (P0, P1, fine grid). Whether a diagonal or low-rank-plus-isotropic IBW would
  remove it is untested.
* **The selected targets are early-stopped transients** (25 iterations), chosen on 40 validation
  pairs from 4 trajectories; converged fits scored worse. Their quality depends on the init seed,
  which is why stage B is compared with an independent replicate fit, not with a second sample set
  of the same fit.
* **Metric reach.** DINOv2 on Tweedie-denoised states is not a fidelity measure at j0–j2 (the
  teacher's own states denoise to colour noise there); 8×-pooled pixel MMD has little power at high
  noise. Each conditional comparison uses 32 vs 32 + 32 samples, so a non-significant test is not
  equality. FID at n = 1,024 carries a common upward bias (real-vs-real 15.1); only paired
  differences between methods on the same 1,024 x_T are interpreted.
* **Scope.** One teacher (CelebA-HQ 256, VE, predictor-only SMLD), one partition (10 × 200 steps),
  K = 8, one refinement round (32 trajectories), one endpoint budget for the ablation (E = 4).
  GPUs 0 and 1 were shared with other users' jobs, so wall-clock costs include contention.
* **Precision.** IBW fits and the student ran with TF32 (user decision); every teacher reference,
  endpoint and evaluation feature is fp32. TF32 does not matter for the targets: refitting 20 test
  pairs (2 trajectories × 10 intervals) in fp32 with identical init, noise and evaluation seeds
  (`frozen_fp32`, `gmmd/exp5/diag_precision.py`, `figures/diag/precision_check_test.json`) moves the
  component means by 0.0001–0.0008 component SDs (an independent replicate's centroid moves
  0.0007–0.15 SDs), leaves the variances unchanged to 2 × 10⁻⁴, and changes the stage-A excess by
  −0.0003 ± 0.0008 (DINOv2; mean excess 0.46) and −0.00002 ± 0.00006 (pixel; mean 0.035).


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
| diagnostics | `$M generate --method ibwchain --n 64`; `$M generate --method hybrid:<a>:<b>:<mask> --n 192`; `python -m gmmd.exp5.diag_detail --split {val,test} --students ...`; `$M figtraj --method student:main`; `python -m gmmd.exp5.diag --split val --its 25 50 300` | online-IBW chain, hybrid samplers, per-interval detail, trajectories, IBW snapshots |
| post-hoc diagnostics (labelled) | `bash run/exp5_hybrid_ablation.sh`; `python -m gmmd.exp5.diag_nll --split test`; `python -m gmmd.exp5.diag_terminal`; `python -m gmmd.exp5.diag_precision`; `$M generate --method student:<tag>:best:meanlast --n 1024` | ablation/main hybrids by interval block, held-out likelihood of the teacher's transitions, terminal spread, TF32 check, noise-free last step |
| tables, costs, figures | `$M tables`, `$M costs --student main` (or `main refined` to charge the refinement), `python -m gmmd.exp5.figures` | |

The full main run as executed: `run/exp5_main.sh` (both chains; GPU0's chain was handed over to
`run/exp5_main_part2.sh` after the main student's shift — see RUNLOG).
