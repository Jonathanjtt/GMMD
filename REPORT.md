# H1: is `p_θ(x_s | x_t)` multimodal?

This report contains the original single-state diagnostic and three focused follow-up
experiments. The follow-up results below are the current summary; the original analysis is
retained afterward for context and provenance.

## Extended Experiment 1/3 continuation (2026-10-17)

The continuation increases evaluation power, measures a teacher-versus-teacher floor, separates
initialisation, capacity, gradient-sample budget and optimisation horizon, and expands the
transition-length study to eight fixed noisy states. It reuses the same teacher, DINOv2
representation, MMD²/SWD definitions and IBW implementation. No conditional neural network is
introduced.

### Experiment 3 — capacity and optimisation

A fresh pool of 256 teacher rollouts (seeds 41000–41255) is disjoint from the 64 samples used for
the oracle initialisation diagnostic. Fits are evaluated against 128 held-out teacher samples.
Ten balanced 128-versus-128 teacher splits give a DINOv2 MMD² floor of
`0.00004 ± 0.00202` (range `-0.00242` to `0.00438`) and SWD `0.1474 ± 0.0136`.

| setting | DINOv2 MMD² | SWD | fit time |
|---|---:|---:|---:|
| standard init, K=1, BG=4, 300 × 0.1 | 0.41756 | 1.02431 | 29.5 s |
| standard init, K=16, BG=4, 300 × 0.1 | 0.39263 | 0.94155 | 502.4 s |
| oracle init, K=2, BG=4, 300 × 0.1 | 0.16673 | 0.64958 | 60.1 s |
| oracle init, K=4, BG=4, 300 × 0.1 | 0.15859 | 0.61250 | 124.5 s |
| oracle init, K=8, BG=4, 300 × 0.1 | 0.15443 | 0.57234 | 249.2 s |
| oracle init, K=16, BG=4, 300 × 0.1 | 0.16313 | 0.63706 | 502.1 s |
| oracle init, K=4, BG=16, 300 × 0.1 (3 seeds) | 0.16891 ± 0.00591 | 0.65591 ± 0.02428 | 498.4 s mean |
| oracle init, K=4, BG=64, 300 × 0.1 | 0.17489 | 0.71844 | 2000.5 s |
| oracle init, K=8, BG=4, 400 × 0.01 (3 seeds) | **0.03895 ± 0.00249** | **0.33736 ± 0.01485** | 330.8 s mean |
| oracle init, K=8, BG=4, 600 × 0.01 | 0.06637 | 0.42837 | 501.3 s |
| oracle init, K=8, BG=4, 1000 × 0.01 | 0.13400 | 0.57131 | 832.5 s |

**What improved.** Oracle placement is by far the largest gain. Once components start in
different semantic basins, K=2 cuts MMD² by about 60% relative to standard K=1; K=4 and K=8 give
smaller further gains. A smaller step size plus early stopping is the second major gain. Along
the common-seed K=8, BG=4, step-0.01 path, MMD² is 0.31198, 0.22242, 0.11378, 0.06404, 0.03831,
0.06637 and 0.13400 at 50, 100, 200, 300, 400, 600 and 1000 iterations, respectively. The
selected 400-step point repeats at 0.03684 and 0.04169 for seeds 29 and 47, giving
`0.03895 ± 0.00249` across three seeds. This is a 90.7% reduction from standard K=1, although
it remains above 0.01 and the teacher null scale.

**What did not.** Standard-init capacity is almost flat from K=2 through K=16 while runtime grows
roughly linearly. Oracle K=16 is worse than K=8. Increasing BG from 4 to 16 or 64 reduces the
late gradient-noise scale but does not improve held-out MMD² or SWD and costs 4×/16× as much.
Continuing K=8 at step 0.1 to 1000 iterations worsens MMD² (0.17536), and even the small-step
trajectory degrades after 400 iterations. More optimization is therefore not monotonically
better under the evaluation metric.

**Bottleneck diagnosis.** Initialisation/local basin discovery is the first bottleneck, but not
the residual one. At the 400-step optimum, mean component distance is 48.3 and normalized mean
separation is 21.9; at 1000 steps these contract to 25.6 and 11.8 while all eight components remain
separated. Thus the U-shaped MMD curve is objective-driven over-contraction, not numerical failure
or mixture collapse. IBW descends reverse KL for the continuous-time Bayes conditional, whereas
evaluation uses samples from a discretized reverse-SDE transition and DINOv2 MMD/SWD.

The variational family is also restrictive. It fixes uniform weights and one isotropic pixel-space
variance per component. The K=8 discovery clusters contain `[2,15,9,6,1,17,13,1]` of the 64
teacher samples, versus eight each under uniform weights (empirical total-variation gap 0.344).
This makes mixture weights a concrete capacity hypothesis; anisotropic covariance is the next one.
The 400-step point was selected using the same 128-sample reference used for its exploratory
metric, so the three-seed result is strong evidence of repeatability, not yet an unbiased final
test estimate.

### Experiment 1 — eight-state combined result

Five new fixed noisy states use 128 paired rollouts per endpoint and fresh seeds; combined with
the original cohort this gives eight states. Negative `BIC(K=2)-BIC(K=1)` favors K=2.

| transition length | states favoring K=2 | mean ΔBIC ± SD | median ΔBIC [IQR] |
|---:|---:|---:|---:|
| 0.02 | 1/8 | 27.74 ± 19.89 | 30.06 [20.68, 43.57] |
| 0.10 | 1/8 | 23.40 ± 49.68 | 36.59 [28.83, 41.93] |
| 0.15 | 3/8 | 4.84 ± 58.57 | 18.50 [-13.93, 46.57] |
| 0.25 | 4/8 | -10.91 ± 67.72 | 7.67 [-32.96, 26.28] |

The prevalence trend supports the original claim—evidence for a two-component alternative is
more common for longer transitions—but it is not universal or monotone within every state. Three
states remain K=1 throughout, two develop evidence only at longer endpoints, and one state is
already strongly multimodal at the shortest endpoint. The mean is influenced by this last state,
so the report retains individual trajectories, the median/IQR and Wilson intervals rather than
claiming a population effect from eight states.

### Prioritized next optimization steps

These steps deliberately stay within the existing GMM/IBW pipeline; the conditional neural
network remains out of scope.

1. **Lock an unbiased early-stopping estimate.** Use one 128-sample half of the existing
   256-teacher pool only for checkpoint selection, then evaluate the selected checkpoint once on
   the untouched half with at least three independent mixture-sampling seeds. This needs no new
   teacher rollouts and prevents selection on the reported test metric.
2. **Test weights before adding more components.** Re-sample the cached 400-step K=8 checkpoints
   using teacher-cluster frequencies as fixed diagnostic weights, then compare with the current
   uniform 1/K sampler under identical evaluation seeds. If this materially closes the gap,
   implement optimized mixture weights. Also run the cheaper K=4, BG=4, step-0.01 early-stopping
   path; its discovery clusters are less imbalanced and it costs about half as much as K=8.
3. **Separate family capacity from score optimization.** Fit a direct teacher-sample isotropic
   mixture with learned weights as a non-IBW upper-bound control. If that control approaches 0.01
   while IBW does not, optimization/objective mismatch dominates; if it also plateaus, extend the
   family to channelwise/diagonal or low-rank-plus-isotropic covariance before revisiting K.
4. **Measure continuous-target versus discrete-teacher mismatch.** Repeat the teacher transition
   with a refined reverse-SDE discretization and compare it with the current teacher at the same
   128–256-sample DINOv2 MMD/SWD scale. A cross-integrator discrepancy above the teacher–teacher
   floor would quantify how much of the IBW gap cannot be removed by optimizing the present
   continuous target more accurately.
5. **Deprioritize brute-force BG/K sweeps.** BG=64 costs about 16× BG=4 without improving either
   metric, and K=16 is already worse than K=8. Further BG or K expansion is justified only after
   the weight, covariance and target-mismatch controls.

For fair comparisons, future rows should keep the fixed noisy state, DINOv2 preprocessing,
128-sample evaluation size, metric seeds and teacher reference split constant, and report both
MMD² and SWD mean ± SD together with fit runtime and score-network evaluations.

Continuation artifacts are in `outputs/exp3_ibw_capacity`,
`outputs/exp1_transition_multimodality_expanded`, and
`outputs/exp1_transition_multimodality_combined`.

## Follow-up experiments 1–3

### Overall conclusion

The results support the project premise, with important limits. Conditional multimodality is
present for some fixed noisy states and becomes easier to detect over longer transitions
(Experiment 1). In a controlled problem, the benefit of a mixture increases sharply with true
mode separation (Experiment 2). On the real high-dimensional conditional, however, standard IBW
does not discover that benefit from a natural initialisation; an oracle initialisation helps
substantially but remains an inadequate fit (Experiment 3).

### Experiment 1 — transition length and multimodality

Three fixed `x_t` states were each evaluated at four endpoints with 64 paired reverse-SDE
rollouts. DINOv2 ViT-B/14 was fixed before analysis; negative `BIC(2)-BIC(1)` favors two
components.

| transition length | result |
| --- | --- |
| `Δt = 0.02, 0.10` | No state favored K=2 over K=1 by BIC. |
| `Δt ≈ 0.15` | One of three states favored a mixture strongly (`ΔBIC = -39.85`). |
| `Δt ≈ 0.25` | Two of three favored a mixture: one strongly (`-49.85`) and one marginally (`-3.08`); the third favored K=1 (`+18.78`). |

**Interpretation.** Longer transitions can expose multimodal conditional structure, but the
effect is state-dependent rather than universal. This is usable as pilot evidence, not an
estimate of prevalence: only three states were tested, cluster-stability evidence was modest,
and one short-transition condition anomalously preferred K=4 even though K=2 lost to K=1.

### Experiment 2 — controlled test of the mixture hypothesis

For a symmetric two-dimensional Gaussian-mixture target, mode separation was swept from 0 to 6.
The reported advantage is `SWD(K=1)-SWD(K=2)`, so positive values favor the mixture.

| mode separation | direct full-covariance | isotropic IBW |
| ---: | ---: | ---: |
| 0 | -0.006 | -0.000 |
| 1 | -0.005 | 0.007 |
| 2 | 0.012 | 0.090 |
| 4 | 0.157 | 0.479 |
| 6 | 0.406 | 1.832 |

**Interpretation.** K=2 provides no spurious benefit for a unimodal or heavily overlapping
target, then becomes progressively better as the modes separate. This cleanly validates the
mechanism behind H1. The two columns should not be compared as equivalent estimators: the direct
K=1 fit has full covariance, whereas IBW K=1 is isotropic and becomes mode-seeking at large
separation. The experiment is an exact low-dimensional control, not evidence that real diffusion
conditionals have the same geometry.

### Experiment 3 — IBW initialisation diagnostic

The canonical real conditional was fitted under three random seeds. Lower MMD² and SWD are
better; teacher-cluster initialisation is an oracle diagnostic, not a practical algorithm.

| condition | mean DINOv2 MMD² | mean SWD | normalized separation |
| --- | ---: | ---: | ---: |
| K=1, standard | 0.4396 | 0.957 | — |
| K=2, standard | 0.4413 | 1.019 | 2.49 |
| K=4, standard | 0.4361 | 0.969 | 2.84 |
| K=2, deliberately spread | 0.4252 | 0.992 | 3.55 |
| K=2, teacher-cluster oracle | **0.2154** | **0.722** | **9.51** |

**Interpretation.** Adding components under standard or merely spread initialisation does not
improve the fit. Initialising K=2 near the teacher clusters reduces MMD² by about 51% and SWD by
about 25% relative to standard K=1, consistently across seeds. Thus initialisation/local
optimisation is a real bottleneck. It is not the only bottleneck: every configuration is still
rejected by the MMD test (`p ≈ 0.005`), and the oracle uses unavailable teacher information.
Isotropic covariance and the continuous-target/discrete-sampler gap remain plausible sources of
the residual mismatch.

### Next steps

1. **Establish prevalence.** Repeat Experiment 1 on more images and fixed `x_t` states, then
   rerun positive conditions with fresh rollout seeds and inspect cluster-conditioned samples.
2. **Test model capacity.** Replace isotropic components with low-rank-plus-diagonal or other
   structured covariances and compare them under the same evaluation protocol.
3. **Separate optimisation from target mismatch.** Increase IBW iterations and gradient samples,
   sweep step sizes, and compare continuous-condition targets with a matched discretised target.
4. **Make mode discovery practical.** Evaluate non-oracle initialisation or component birth/split
   schemes against the teacher-cluster upper-bound diagnostic.

Detailed artifacts are in `outputs/exp1_transition_multimodality`, `outputs/exp2_toy_h1`, and
`outputs/exp3_ibw_initialization`; each directory contains a `SUMMARY.md`, numerical outputs,
figures, resolved configuration, and provenance.

---

## Original single-state diagnostic

Reproduced by the commands in each run's `provenance.json`; every number below is read from
`results/*/diagnostics.json` and `results/*/representation_check.json`.

---

## The question, and why the obvious answer is the wrong one

CelebA-HQ is multimodal: it contains many different faces. That is uninteresting. The question
here is about the **conditional**. Fix one noisy tensor `x_t`, then run the reverse SDE from it
many times. Does the teacher's own stochasticity carry the sample to genuinely *distinct*
outcomes, or to one broad smear around a single answer?

The distinction matters because it is the premise of the whole project. If `p_θ(x_s | x_t)` is
unimodal, a single Gaussian variational approximation loses nothing and there is no case for a
mixture. If it is multimodal, there is.

## Setup

| | |
| --- | --- |
| Teacher | Song et al. score-SDE, CelebA-HQ 256 NCSN++, VE SDE, EMA weights, checkpoint step 2 400 005 |
| Image | `validation-00000-of-00001.parquet#row0`, one held-out CelebA-HQ 256 image |
| `x_t` | drawn **once** with forward-noise seed 0, then frozen; SHA-256 `508d4c129745…` |
| Long transition | grid 800 → 1099, **299 steps**, `t = 0.5998` (σ 5.30) → `s = 0.4502` (σ 1.11) |
| Short control | grid 800 → 840, **40 steps**, → `s_near = 0.5798` (σ 4.30) |
| Samples | 64 independent rollouts, seeds 1000–1063, one noise stream each |
| Samplers | (a) reverse-diffusion predictor, no corrector — the plain reverse SDE; (b) the same plus the official Langevin corrector, snr 0.075 |
| Deterministic control | probability-flow ODE from the same `x_t`, 3 repeats |

The short control is not a separate set of runs. It is the *same* 64 rollouts snapshotted at step
40, which is bit-identical to running them only that far, so the two differ solely by transition
length and share their random streams.

---

## Result 1 — the spread is real

Three probability-flow ODE runs from the same `x_t` agreed to `max |diff| = 0.000e+00`. Exactly
zero, not merely small. So the fixed-`x_t` plumbing does not leak, the network is deterministic
under repetition, and **every bit of the observed spread is reverse-SDE noise.**

## Result 2 — pixel space says nothing, and that is a measurement failure

The first pass scored everything in 8× average-pooled pixels, as the config specified. It found
nothing: best `K = 1` by BIC and by held-out likelihood, in every group, under both samplers, with
an unstable 2-means split (ARI 0.49, p = 0.149).

That conclusion does not survive looking at the samples. `results/conditional_multimodality/sample_grid.png`
shows 64 rollouts from one frozen `x_t` that are plainly **different people**. Pixel distance was
not measuring the thing the question is about: two faces at the same pose and lighting are close
in pixel space regardless of whose faces they are. Only 45% of the variance survived the
projection to 10 principal components.

`gmmd.experiments.representation_check` re-scores the identical saved samples under DINOv2
ViT-B/14, repeating no rollouts.

## Result 3 — under a semantic representation, the plain reverse SDE is bimodal

| Sampler | Group | Best K (BIC) | Best K (held-out) | ΔBIC, K=2 vs K=1 | Cluster stability ARI (p) |
| --- | --- | --- | --- | --- | --- |
| reverse SDE | **long** | **2** | **2** | **−41.3** | **0.96 (0.020)** |
| reverse SDE | short control | 1 | 1 | +36.8 | 0.70 (0.327) |
| PC + Langevin | long | 1 | 2 | +0.9 | 0.99 (0.010) |
| PC + Langevin | short control | 1 | 2 | +5.9 | 0.92 (0.010) |

For the plain reverse SDE the separation is clean and every diagnostic agrees. Two components beat
one by 41 BIC units on the long transition and *lose* by 37 on the short one. Held-out likelihood
picks the same winners. A bootstrap adjusted Rand index of 0.96 says the two-cluster partition is
essentially perfectly reproducible under resampling, against 0.70 for the control, which is not
distinguishable from a unimodal Gaussian null (p = 0.327).

This is the shape of result H1 asks for: the treatment separates, the control does not.

## Result 4 — the two modes are interpretable

The 45/19 split of the long transition is not a statistical curiosity.
`results/conditional_multimodality/clusters_dinov2_vitb14.png` renders it: **cluster 0 is
male-presenting faces, cluster 1 is female-presenting.** From one fixed noisy tensor at σ ≈ 5.3,
the reverse SDE lands in one of two semantically distinct basins.

That is also the reason pixel space missed it. The pixel and DINOv2 partitions agree at only
ARI 0.585, and the pixel one does not reproduce, so pixel distance was tracking something else,
most plausibly overall brightness and background.

## Result 5 — the corrector destroys the control

The official predictor–corrector sampler shows a stable two-cluster split on the long transition
(ARI 0.99) **and on the 40-step short control** (ARI 0.92, p = 0.010). Its control does not behave
like a control.

The Langevin corrector injects extra noise at every noise level in order to re-target the
marginal. It roughly doubles the short transition's spread (mean pairwise DINOv2 distance 32.7
against 26.4) and the sample cloud has already split in two after 40 steps. It also inflates
within-cluster variance enough that BIC no longer pays for the second component even where
k-means finds it reliably, which is why its BIC and held-out columns disagree.

**The two samplers therefore define different conditionals, and the reported multimodality of the
long transition is a property of the plain reverse SDE.** Neither is declared correct here. The
plain reverse SDE is the one whose transition kernel matches the object the question names, so it
carries the headline result; the corrector's behaviour is reported rather than averaged away.

---

## What this does and does not establish

**Supported.** For this image, this `x_t` and this `(t, s)`, the plain reverse-SDE conditional
`p_θ(x_s | x_t)` is better described by two components than by one, the split is semantically
coherent, and a much shorter transition from the same `x_t` is not.

**Not supported, and the first one is a real weakness.**

1. **The representation was chosen after seeing a null result.** The pixel extractor was what the
   config named; DINOv2 was tried after it returned nothing. That is a forking path. The
   short-transition control under the *same* representation is what keeps it honest, since a
   representation that manufactured structure would manufacture it in the control too. It is still
   a hypothesis-generating result. The clean version is to fix DINOv2 in the config and re-run on
   fresh rollout seeds, about 12 GPU-minutes.
2. **One image, one `x_t`, one `(t, s)` pair, 64 samples.** Nothing here generalises to diffusion
   conditionals at large, or even to other noise levels.
3. **The bootstrap likelihood-ratio test should be ignored.** It returns p = 0.005 for almost every
   group including both controls. It detects non-Gaussianity, not multimodality. It is reported
   because it was pre-specified, and it is discounted because the control shows it is not
   specific.
4. **"Two modes" is the model order that wins among K ≤ 4 under a diagonal-covariance mixture in a
   10-dimensional PCA subspace.** It is not a claim that the true conditional has exactly two.
5. Diagnostics run on the teacher's own denoised view `E_θ[x₀ | x_s]`. Denoising is deterministic,
   so it cannot invent modes, but it does change the metric.

---

# IBW against the conditional: how many components?

The target is the closed-form conditional score of the continuous-time reversal,
`s_θ(x_s, s) + (x_t − x_s)/(σ_t² − σ_s²)`, which for a VE SDE is exact given the score network.
It is **not** the discretised sampler's kernel, which has no score. Its log-density is unavailable,
so the objective can be descended but not evaluated, and model order is chosen by an MMD
permutation test of the fitted `q`'s samples against the teacher's own 64 conditional samples.
See [`gmmd/conditional_target.py`](gmmd/conditional_target.py).

Two convergence checks hold everywhere below: the mean-gradient norm falls by two orders of
magnitude, and the fitted variance matches the analytic posterior variance a Gaussian marginal
would give, to 0.2% (short: 6.3284 against 6.3146; long: 1.1775 against 1.1751).

## Short transition: K = 1, and IBW merges the rest by itself

| K | MMD² vs teacher (DINOv2) | p | MMD² (pixel) | p | component separation / σ |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.01216 | 0.030 | −0.00002 | **0.529** | — |
| 2 | 0.01126 | 0.016 | +0.00007 | 0.367 | 2.03 |
| 4 | 0.01130 | 0.016 | +0.00007 | 0.373 | 2.19 |
| 8 | 0.01144 | 0.014 | +0.00007 | 0.369 | 2.34 |

One component already passes in pixel space, extra components change nothing, and IBW collapses
them on its own: initialised ~600 σ apart, which is simply what independent draws are at
d = 196 608, they finish 2 σ apart with identical variances to four decimals. That reproduces H1's
verdict — the 40-step transition is unimodal — from the optimiser rather than from clustering.

## Long transition: no K works from the natural initialisation

| K | MMD² (DINOv2), prior-free init | MMD² (DINOv2), started at the teacher's modes |
| --- | --- | --- |
| 1 | 0.4612 | 0.3452 |
| 2 | 0.4699 | **0.2108** |
| 4 | 0.4693 | 0.2113 |
| 8 | 0.4691 | — |

From the natural (prior-free) start, every K is rejected at p = 0.002, at 35× the null, and **K
is irrelevant** — the four values are flat within noise. The components merge exactly as in the
unimodal case, ending 4.4–5.5 σ apart. IBW converges to a single isotropic Gaussian however many
components it is given.

This is the locality of Bures–Wasserstein transport: a W-type flow moves components downhill
locally and cannot carry mass across a low-density valley, so each one falls into whichever basin
it started nearest. It is the same phenomenon the sibling GMMVI project exists to address.

## The collapse is an optimisation failure, not a unimodal target

Restarting the components **on** the teacher's own modes — the k-means representatives in DINOv2
space, with the variance the free fit converged to, so the only change is where they start —
separates the two explanations:

* **K = 2 now beats K = 1**, 0.2108 against 0.3452, a 39% reduction. From the prior-free start the
  two were indistinguishable. So the target does carry structure that a second component captures.
* **Initialisation changes the answer even at K = 1** (0.3452 against 0.4612), so the objective is
  non-convex and the fits below are local optima, not the global one.
* The components stay much further apart, 18.3 σ against 5.5 σ, though still far from their
  638 σ start.

So the conditional that IBW is optimising is not unimodal; plain IBW simply cannot find its
second mode from a naive start. That is positive evidence for this project's premise — a mixture
does beat a single Gaussian — obtained without assuming it.

## What is still unexplained

Even the best configuration is rejected decisively: K = 2 with mode initialisation sits at 0.2108
against a null of 0.0108, twenty times over. So an isotropic mixture, however initialised and
however many components, does not reproduce this conditional. Two candidates remain, and nothing
here separates them:

1. **Isotropy.** The true conditional's covariance is presumably anisotropic, stretched along the
   data manifold; spherical components cannot represent that at any K. Note the short-transition
   fit failed in DINOv2 while passing in pixels, which already pointed this way.
2. **The discretisation gap.** The target is the continuous-time conditional; the teacher samples
   come from a 299-step discretisation of it, and a corrector would change it again.

A further caution specific to this dimension: two isotropic components 18 σ apart still overlap
almost completely, because samples sit on a shell of radius √d σ ≈ 443 σ around each mean.
"Separated modes" in raw pixel space at d = 196 608 does not mean what it means in two dimensions,
and the component-separation column should be read as a relative diagnostic only.

## Next

In order of value: fix the representation in the config and re-run H1 on fresh seeds, since the
DINOv2 choice was made after seeing a null; test full-covariance or low-rank components against
the isotropy hypothesis; measure the discretisation gap directly by fitting the same target at
matched step counts; sweep `(t, s)`; repeat on several `x_t` and several images.
