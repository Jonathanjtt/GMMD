# exp5 — design record (written before any large run)

Research question: can our existing IBW conditional-mixture fitting be amortised into a neural
network, so that ten 8-component mixture draws replace the complete diffusion sampling trajectory
with useful fidelity and lower inference cost?

Code: `gmmd/exp5/` (grid, ibw_targets, student, sampler, metrics, train, evaluation),
driver `gmmd/experiments/ibw_amortized.py`, config `configs/exp5_ibw_amortized_10step_k8.toml`,
tests `tests/test_exp5.py`. Branch `exp5-ibw-amortized`.

## 1. What the existing IBW implementation is (inspected, not assumed)

| question | answer (with source) |
|---|---|
| Distribution targeted | The **continuous-time VE Bayes conditional** `p(y | x) ∝ p_s(y) · N(x; y, (σ_t² − σ_s²) I)` of the time reversal under the learned marginal score (`gmmd/conditional_target.py`, batched in `gmmd/ibw_batched.VEConditionalScores`). |
| How the gradient-flow information is obtained | Score only: `∇_y log p(y | x) = s_θ(y, s) + (x − y)/(σ_t² − σ_s²)` — the score network evaluated at the END time `s`, plus the exact VE forward-kernel term. The log-density is unavailable (needs `log p_s`), so the reverse-KL objective can be descended but not evaluated. |
| Teacher's multi-step kernel, or an approximation? | **An approximation.** The teacher's interval transition is a composition of 200 SMLD Gaussian steps with no closed-form score. The Bayes conditional is the continuous-time object that the SMLD kernel converges to as the grid is refined (checked on the analytic toy teacher, `tests/test_teacher_toy.py::test_*refined*`). It is exact for the true time reversal only if `s_θ` were the exact score of `p_s` (a learned field need not be conservative). The marginal score is **not** used as the transition score; the Bayes correction term is what makes it a conditional. No claim of exact transition matching is made: stage A measures the gap against teacher rollouts. |
| Parameters updated | Means by the (GD) step and isotropic variances by IBW Eq. (12) (`gmmd/ibw.py`, `gmmd/ibw_gpu.py`, `gmmd/ibw_batched.py`). **Weights are fixed at 1/K** and never learned (the family the paper's updates are derived for). |
| Covariance parameterisations | **Isotropic only**: one scalar variance per component, `Σ_k = ε_k I`. Dense covariances exist only as a read-only view for d ≤ 4096 in the NumPy reference; nothing optimises diagonal, low-rank or full covariances. At d = 196 608 we therefore use the isotropic family, with no projection or substitute flow. |

Consequences for the student: it uses the **same isotropic family** per component (one scalar
variance per component, global weights over the whole image) plus learned weights, which
contain IBW's uniform weights as a special case. No extra covariance approximation is
introduced at the amortisation stage.

## 2. Ten intervals over the complete teacher trajectory

Teacher frozen: Song et al. CelebA-HQ 256 NCSN++ (VE SDE, EMA weights, `checkpoint_48.pth`),
the exp4 "original pipeline": predictor-only reverse diffusion (SMLD), no corrector, official
grid `linspace(1, 1e-5, 2000)`, prior `x_T = 348 z`, noise removal at the end (output = the last
predictor mean, float images ≈ [0, 1]). The grid is unchanged: 2,000 steps → 10 × 200.

| j | teacher steps | grid start → end state | t start → end | σ start → end | gap σ_t²−σ_s² | v_anc | mean gap |
|---|---|---|---|---|---|---|---|
| 0 | [0, 200) = 200 | 0 → 200 | 1.00000 → 0.89995 | 348 → 122.2 | 1.062e+05 | 1.31e+04 | 1.062e+05 |
| 1 | [200, 400) = 200 | 200 → 400 | 0.89995 → 0.79990 | 122.2 → 42.94 | 1.31e+04 | 1616 | 1.31e+04 |
| 2 | [400, 600) = 200 | 400 → 600 | 0.79990 → 0.69985 | 42.94 → 15.08 | 1616 | 199.4 | 1616 |
| 3 | [600, 800) = 200 | 600 → 800 | 0.69985 → 0.59980 | 15.08 → 5.297 | 199.4 | 24.6 | 199.4 |
| 4 | [800, 1000) = 200 | 800 → 1000 | 0.59980 → 0.49975 | 5.297 → 1.861 | 24.6 | 3.035 | 24.6 |
| 5 | [1000, 1200) = 200 | 1000 → 1200 | 0.49975 → 0.39971 | 1.861 → 0.6536 | 3.035 | 0.3745 | 3.035 |
| 6 | [1200, 1400) = 200 | 1200 → 1400 | 0.39971 → 0.29966 | 0.6536 → 0.2296 | 0.3745 | 0.0462 | 0.3745 |
| 7 | [1400, 1600) = 200 | 1400 → 1600 | 0.29966 → 0.19961 | 0.2296 → 0.08064 | 0.0462 | 0.0057 | 0.0462 |
| 8 | [1600, 1800) = 200 | 1600 → 1800 | 0.19961 → 0.09956 | 0.08064 → 0.02832 | 0.0057 | 0.0007033 | 0.0057 |
| 9 | [1800, 2000) = 200 | 1800 → 1999 (+ noise removal) | 0.09956 → 0.00001 | 0.02832 → 0.01 → 0 | 0.0007022 | 8.755e-05 | 0.0008022 |

(`partition.json` has the same data plus the discrete SMLD noise levels.) Each interval lowers σ
by ≈ 2.85×; `v_anc = σ_s² · gap / σ_t²` is the Gaussian-data posterior variance.

**Terminal interval (explicit).** Steps [1800, 1999) are stochastic and end at the σ_min = 0.01
state (grid 1999); step 1999 is deterministic under noise removal,
`x_0 = T(x) = x + σ_min² s_θ(x, ε)` (the vendored predictor's mean, `grid.terminal_denoise`).
IBW fits the Bayes conditional of the σ_min state given `x_1800` (score at `t = ε`). The
student's 10th kernel is trained on the **pushforward** `T # q_IBW` — a cached bank of 32
samples `T(z)`, `z ~ q_IBW`, per terminal fit, computed offline (the map needs the network) —
so at inference the 10th draw is already in the teacher's output representation and no teacher
call is made. Stage A for interval 9 compares teacher outputs with `T(z)` samples; the common
post-processing for every method is the upstream quantisation
`clip(x·255, 0, 255).astype(uint8)` before FID/P-R. No other decoding exists (pixel space).

## 3. Offline IBW fits (stage A, offline)

* One fit = one anchor (a teacher-trajectory state at an interval start) × one interval; K = 8,
  B = 4 gradient samples per component, Eq. (12) unchanged, batched over 4 anchors per network
  call (`ibw_batched.fit_batched`, which reproduces `ibw_gpu.fit` anchor by anchor).
* **Step size per interval:** `γ_j = c · v_anc,j`. Because `−∇² log p_s ⪯ I/σ_s²`, the target's
  precision is at most `1/v_anc`, so this gives every interval the same stiff-direction rate `c`
  (stable for c < 2) — only the scalar γ changes; the update equations do not. (exp4 used
  γ = 0.1 at v_anc ≈ 6.3, i.e. c ≈ 0.016, and was still under-converged after 300 iterations.)
  `c` is set by the P0 check (exploratory; convergence diagnostics + DINOv2 excess).
* **Iterations** (≠ diffusion steps): max 400, snapshots 25/50/100/200/300/400; the count is
  chosen on validation anchors by the pre-registered rule below. **Convergence diagnostic**
  (reported for every fit): converged ⇔ over the last 25 iterations the mean drift
  `‖m(T) − m(T−25)‖_F / √(K d mean v) < 0.02` and `max_k |Δ log v_k| < 0.01`, and no early stop.
* **Initialisation (user decision: K = 8, teacher-sample-free, components far apart; initial
  mean and covariance tuned on validation):** variants `prior_free` (repository default),
  `principal_r1`, `principal_r3`, `principal_r3_vgap` (± along the top 4 eigenvectors of the
  Tweedie posterior covariance `gap (I + gap J)`, `J = ∂s_θ(x, t)/∂x`, around the posterior mean
  `x + gap s_θ(x, t)`, at 1 or 3 posterior SDs, initial variance v_anc or gap), and `random_far`
  (random unit directions at radius √(d v_anc) — the literal far spread, as a control). The
  principal directions come from 8 subspace iterations with central finite differences in full
  fp32 (≈ 100 score evaluations per fit; TF32 rounding exceeds the difference).
* Recorded per fit (`fits/<split>/<variant>/traj###_j#.{npz,json}`): anchor, split, x_T seed,
  interval (indices, times, σ), init (scheme, seeds, eigenvalues, component stats), noise seed,
  γ, iterations, gradient-norm trace, variance and mean-drift traces, convergence flags,
  early-stop messages, effective components (single-linkage merge at separation < 1 σ), variance
  ratios to v_anc, NFE, runtime, teacher checkpoint SHA-256 and sampler identifiers, precision,
  git commit. Fits are never discarded; failures stay in every summary.

## 4. Splits, seeds, precision

* Independent parent trajectories per split (x_T seed = base + n; base 3.0M train, 3.1M val,
  3.2M test, 3.3M refinement; exp4's FID seeds are 1.0M+). Manifests in `manifests/`, written
  before any target. 64 train, 4 val (selection, early stopping), 8 test (all held-out metrics).
* Precision (user decision): teacher references fp32/TF32-off; IBW fits, banks and the student
  TF32; principal directions fp32.

## 5. Student (stage B)

NCSN++ initialised from the teacher's EMA weights (65.6 M parameters); the 7 output-skip convs
widened 3 → 24 channels (8 components, tiled kernels + 1 % perturbation, component 0 exact);
global head (pooled 4×4 bottleneck + interval embedding → 8 logits + 8 log-variance offsets,
zero-initialised). `μ_k = x + g_j h_k(x, σ_j)`, `log v_k = log v_ref,j + 6 tanh(δ_k/6)`. At
initialisation every component is the teacher's first-order Tweedie mean with variance v_ref:
the student starts as a 10-step ancestral-like sampler and must learn whatever IBW adds.
Loss: `E_(x,j) E_{z ~ q_IBW} [−log q_φ(z | x, j)]` with fresh draws each step (bank rows for
the terminal interval), per-component log densities summed over all coordinates before the
log-sum-exp, standardised per interval. Adam(W), EMA 0.999, val NLL on held-out val fits.

## 6. Pre-registered selection rule (copied from the config, fixed before P1 results)

For every validation (trajectory, interval) pair and every (variant, snapshot):
`X = MMD²_u(IBW, A) − MMD²_u(B, A)` on DINOv2 features of Tweedie-denoised states (A, B: two
independent 32-sample teacher sets; bandwidth = median heuristic on A ∪ B). A failed fit scores
the maximum attainable excess (2). (1) Variant: lowest mean X over the 40 val pairs at its best
snapshot; if the paired-bootstrap 95 % CI of X[variant] − X[prior_free] contains 0, the tie
between the two is broken by failure rate, then mean effective components. (2) Iterations: the
smallest snapshot whose paired-bootstrap CI of X(it) − X(best it) has lower bound ≤ 0 with no
higher failure rate.

## 7. Evaluation

Stage A (teacher → IBW), B (IBW → student: forward/reverse KL per dimension between the two
mixtures, NLL, sample MMD against an IBW-vs-IBW floor), C (teacher → student) at the 8 held-out
test anchors × 10 intervals, with independent evaluation-only IBW fits and 32-sample teacher
controls; DINOv2 + pixel. Rollout shift at the boundaries; refinement on student-rollout anchors
if needed. Baselines: full teacher (exp4's 1,024 fp32 images, same x_T seeds → paired FID
bootstrap), coarse teachers with 10 network evaluations (SMLD reverse diffusion and ancestral;
plus ancestral 20/50/100 for a quality–NFE curve), and the direct-NLL-on-teacher-endpoints
ablation. FID/P-R at identical n and preprocessing against all 30k CelebA-HQ 256 images;
uncached end-to-end timing after warm-up, synchronised; offline costs reported separately.
