# H1: is `p_θ(x_s | x_t)` multimodal?

**A first diagnostic experiment on one fixed `x_t`.** Reproduced by the commands in each run's
`provenance.json`; every number below is read from `results/*/diagnostics.json` and
`results/*/representation_check.json`.

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
