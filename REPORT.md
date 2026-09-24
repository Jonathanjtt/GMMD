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

## Next

The obvious follow-ups, in order of value: fix the representation in the config and re-run on
fresh seeds; sweep `(t, s)` to find where the conditional stops being bimodal; repeat on several
`x_t` and several images. Only then does the comparison this project exists for — one Gaussian
against an IBW mixture — mean anything.

Fitting that mixture is still blocked on a separate problem: reverse-KL variational inference
needs `∇_{x_s} log p_θ(x_s | x_t)`, and the teacher supplies only the marginal score plus a
sampler. See the module docstring of [`gmmd/vi.py`](gmmd/vi.py).
