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

_(sections 2–8 are filled from the result files at the end of the run)_
