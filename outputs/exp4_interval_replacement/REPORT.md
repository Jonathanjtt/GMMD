# exp4 — Replacing one sampling interval with GMMD

**Question.** With the same pretrained score-SDE model (Song et al., CelebA-HQ 256), how does
replacing one sampling interval with a GMMD (IBW Gaussian-mixture) fit change final image
quality and actual computational cost?

**Short answer (preliminary: 192 images per method).** The interval that GMMD approximates best is
a short one: 40 of the 2,000 sampler steps. Replacing it gave a final FID indistinguishable from
the original sampler. But it made each image **58% more expensive** (116 s vs 73 s;
3,160 vs 2,000 score evaluations), because the per-image IBW fit costs 30× the segment it
replaces. A coarse teacher using 4 steps instead of 40 also matched the original FID, at
1/300 of GMMD's segment cost. On this interval GMMD therefore offers no quality–cost advantage.
Its conditional samples are also detectably different from the teacher's at every validation
anchor, while the FID comparison is too weak to detect any effect.

> Scope note. The plan was 1,024 images per method. After the pilot, the run was stopped by
> decision (the experiment is a test run). Every three-way number below uses the **192-image
> pilot**; supplementary FIDs use the extra teacher and coarse images already generated.
> No uncached end-to-end timing was run (see §5).

---

## 1. What the existing implementation is

| | |
|---|---|
| Data / resolution | CelebA-HQ 256×256 RGB, float in [0, 1], not centred, no crop (`gmmd/datasets/celebahq.py`) |
| Checkpoint | `ve/celebahq_256_ncsnpp_continuous/checkpoint_48.pth`, NCSN++ continuous, EMA weights, fixed (no retraining) |
| SDE / schedule | VE SDE, σ(t) = 0.01·(348/0.01)^t, official grid `linspace(1, 1e-5, 2000)`; step *i* uses the score at t_i and moves σ from `discrete_sigmas[k_i]` to `discrete_sigmas[k_i−1]` |
| Sampler ("original pipeline") | Reverse-diffusion (SMLD) predictor, **no corrector**, 2,000 steps, final noise removal (returns the last predictor mean). The official CelebA-HQ default adds a Langevin corrector; predictor-only was chosen to match all earlier GMMD work. With no corrector there is no boundary update to assign. |
| Precision | fp32, TF32 off for all sampling and fitting |
| What GMMD approximates | The continuous-time Bayes conditional p(x_s \| x_t), whose score is s_θ(x_s, s) + (x_t − x_s)/(σ_t² − σ_s²) (`gmmd/conditional_target.py`). This is **not** the discretised sampler's kernel; the gap is measured, not assumed away. |
| Space | **Raw pixel space** (d = 196,608). GMMD samples are diffusion states and can be fed straight back into the sampler; no decoder is involved. DINOv2 is used only to *evaluate*. |
| Family | Uniform-weight mixture of K isotropic Gaussians (one variance per component) |
| Fitting | **One separate IBW fit per x_t** (no conditional network); uses score evaluations only, no teacher samples. Cost = K·B score evaluations per iteration. |

## 2. Interval selection (validation, frozen before generation)

**Candidates.** Only two intervals had earlier GMMD evaluations, and both start at grid
index 800. The earlier one-anchor numbers came from forward-noised real images and a different
MMD bandwidth, so they were not reused as evidence; both candidates were re-validated.

**Protocol (selection rule written into the config before any result existed).**
- 16 fresh anchors. Each x_T came from a seed disjoint from the test seeds, then ran through
  the same teacher prefix [0, 800) used in generation. The long candidate used anchors 0–7.
- At each anchor, two independent sets of 64 teacher samples: A is the reference, B measures
  the teacher-vs-teacher floor. Plus 64 draws from that anchor's own fitted mixture.
- GMMD fitting uses no teacher samples, so fitting and evaluation data are disjoint by construction.
- Metric: unbiased MMD² on DINOv2 features of Tweedie-denoised states, with the Gaussian-kernel
  bandwidth set from teacher A ∪ B. "Excess" = MMD²(GMMD, A) − MMD²(B, A). Also reported:
  SWD (256 projections), pixel-space MMD², and permutation-test detection rates (α = 0.05).
- GMMD config family, fixed in advance from earlier results: K = 1, B = 4, step 0.1,
  prior-free initialisation. Only the iteration count was validated, via snapshots at 100/200/300.

| candidate | grid i_t→i_s | t → s | σ_t → σ_s | steps replaced | iters | anchors | GMMD MMD² | teacher floor MMD² | **excess** (mean ± SD) | SWD GMMD / floor | detected: GMMD / floor (DINOv2) | detected (pixel) | numerical failures | GMMD NFE / anchor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| short | 800→840 | 0.5998→0.5798 | 5.30→4.30 | 40 | 100 | 16 | 1.319 | −0.001 | 1.320 ± 0.082 | 1.54 / 0.12 | 16/16 / 0/16 | 16/16 | 0 | 400 |
| short | 800→840 | | | 40 | 200 | 16 | 0.593 | −0.001 | 0.594 ± 0.117 | 0.71 / 0.12 | 16/16 / 0/16 | 16/16 | 0 | 800 |
| **short** | **800→840** | | | **40** | **300** | **16** | **0.039** | **−0.001 ± 0.002** | **0.040 ± 0.014** | **0.20 / 0.12** | **16/16 / 0/16** | **0/16** | **0** | **1,200** |
| long | 800→1099 | 0.5998→0.4502 | 5.30→1.11 | 299 | 100 | 8 | 1.126 | 0.001 | 1.126 ± 0.160 | 1.73 / 0.20 | 8/8 / 1/8 | 8/8 | 0 | 400 |
| long | 800→1099 | | | 299 | 200 | 8 | 0.267 | 0.001 | 0.266 ± 0.085 | 0.70 / 0.20 | 8/8 / 1/8 | 8/8 | 0 | 800 |
| long | 800→1099 | | | 299 | 300 | 8 | 0.351 | 0.001 | 0.351 ± 0.072 | 0.81 / 0.20 | 8/8 / 1/8 | 8/8 | 0 | 1,200 |

Teacher cost to generate each anchor's reference samples (evaluation only, not part of GMMD's
cost): 128 × 40 = 5,120 score evaluations (329 s) for short and 33,152 (2,132 s) for long.
Validation took 2.7 h of wall-clock on two H200s.

**Frozen selection** (`validation/selection.json`):
- **Interval:** short, grid 800→840, t 0.5998→0.5798, σ 5.30→4.30. This is 40 of 2,000 steps
  (2% of the sampler's score evaluations). Paired excess difference short − long on the 8 shared
  anchors: −0.314, 95% CI [−0.360, −0.276]. Even long's best snapshot (0.266 at 200 iterations)
  is far worse than short.
- **GMMD:** K = 1, B = 4, step 0.1, **300 iterations**, prior-free init N(x_t, (σ_t² − σ_s²) I),
  one random draw per image. 100 and 200 iterations were significantly worse (CIs exclude 0).

Short is a nonzero but brief interval. Success on it is no evidence about long intervals; the
long candidate is ~9× worse.

## 3. Replacement pipeline

```
x_T → teacher prefix [0, 800) → x_t → { A: teacher [800, 840) | B: 4 coarse steps | C: IBW fit + 1 draw } → x_s → teacher suffix [840, 2000) → x_0
```

- **Shared across methods:** x_T, the cached x_t, the suffix configuration and the suffix noise
  streams. Sharing suffix noise (common random numbers) leaves each method's marginal law intact.
- **Not shared:** each method draws its own segment noise. A per-image identity change between
  columns of the comparison figure is ordinary conditional randomness, not an error.
- **Exactness:** the three-way split is bit-identical to the uninterrupted loop; every step runs
  exactly once (`tests/test_interval_replacement.py`). An image's trajectory does not depend on
  its batch.
- **Coarse teacher:** 4 SMLD steps over the same σ range (blocks of 10 fine steps). With one fine
  step per block it reproduces the fine sampler, which is tested.
- **GMMD:** each image gets its own fit, batched 64 images at a time for GPU efficiency. The
  batched fitter reproduces `gmmd.ibw_gpu.fit` image by image (tested). The output is a random
  draw from the image's own mixture, never its mean.
- **Test seeds:** x_T seeds 1,000,000+ (validation used 500,000+). 192 distinct x_T, one image each.

## 4. Results (192 images per method — preliminary)

FID: Inception-v3 (TF-compatible) pool3 features, against all 30,000 CelebA-HQ 256 images
(train and validation). Precision/recall: VGG16 fc2 features, k = 3, against a fixed random
192-image real subset. Both via torch-fidelity, with identical preprocessing for real and
generated images (256×256 uint8, quantised as upstream does). FID ± is the bootstrap SD over images.

| Method | Replaced interval | Segment steps / GMMD configuration | Number of images | FID ↓ | Precision ↑ | Recall ↑ | Segment cost ↓ | Full generation cost ↓ | Peak memory |
|---|---|---|---|---|---|---|---|---|---|
| A. Original teacher | none | 40 fine SMLD steps | 192 | 56.57 ± 2.16 | 0.776 | 0.542 | 1.46 s/img, 40 NFE | 73.2 s/img, 2,000 NFE (49 img/GPU-h) | 48.4 GB |
| B. Coarse teacher | grid 800→840 | 4 SMLD steps | 192 | 57.99 ± 2.15 | 0.719 | 0.458 | 0.15 s/img, 4 NFE | 72.3 s/img, 1,964 NFE (50 img/GPU-h) | 48.4 GB |
| C. GMMD replacement | grid 800→840 | IBW K=1, B=4, 300 it × 0.1, prior-free init, 1 draw | 192 | 58.05 ± 2.30 | 0.786 | 0.500 | 43.75 s/img, 1,200 NFE (fit 43.75 s, sampling < 0.01 s) | 116.0 s/img, 3,160 NFE (31 img/GPU-h) | 49.9 GB |

NFE = score-network examples evaluated per image, not batched calls. Every stage was timed with
the GPU synchronised, at 64 images per network call, on an H200 shared with another user's light
job. Full cost = prefix (29.5 s, 800 NFE) + segment + suffix (≈ 42.5 s, 1,160 NFE).

**Paired FID differences** (same x_T, bootstrap over images): GMMD − teacher, 95% CI [−2.3, +5.1];
coarse − teacher, [−2.2, +5.0].

**Sample-size context (supplementary, `metrics_supplementary.json`):**

| n | teacher FID | real-vs-real FID at same n | teacher P / R | coarse FID | coarse P / R |
|---|---|---|---|---|---|
| 192 | 56.6 | 50.5 | 0.776 / 0.542 | 58.0 | 0.719 / 0.458 |
| 448 | 37.4 | 29.8 | 0.772 / 0.487 | 38.6 (paired Δ CI [−0.8, +3.5]) | 0.732 / 0.442 |
| 1,024 | 22.6 | 15.1 | 0.769 / 0.464 | — | — |

At n = 192 most of the FID is small-sample bias: 192 *real* images score 50.5. The teacher sits a
steady 6.1–7.6 FID above real data at every n.

**Stability:** 0 non-finite values and 0 IBW fits stopped early (192 fits). Final mixture
variance averaged 6.45 (initial 9.60). Outputs fell in [−0.025, 1.12] before quantisation, with
0.7–0.8% of values clipped for every method. Image grids (the first 64, no selection):
`figures/grid_{teacher,coarse,gmmd}_pilot.png`; same-x_T comparison:
`figures/paired_pilot_labeled.png`.

## 5. Answers

**1. Which interval, and why?** The short interval: grid 800→840, 40 steps, t 0.600→0.580,
σ 5.30→4.30. It has much lower excess conditional error than the long interval on the same
anchors (0.040 vs 0.351; paired CI excludes 0). The pre-registered rule then chose 300
iterations, the most tested.

**2. Does the approximation hold across anchors?** Consistently, but not perfectly:
- Excess MMD² was positive at all 16 anchors (range 0.016–0.070), with no numerical failures.
- In DINOv2 space, GMMD was detectably different from the teacher at **16/16** anchors
  (p ≈ 0.002), while the two teacher sets were never told apart (minimum p 0.14).
- SWD was 0.20 vs a teacher floor of 0.12.
- In raw pixels, GMMD was undetectable at all 16 anchors.

So the error sits in semantic structure, not pixel statistics.

**3. Effect on FID and diversity?** None detectable at n = 192. FID: 56.6 teacher, 58.1 GMMD;
paired difference CI [−2.3, +5.1]. Recall was 0.54 vs 0.50 and precision 0.78 vs 0.79, both
within rough sampling error (binomial SE ≈ 0.03–0.04 per value). This is weak evidence. The
coarse teacher, with 10× fewer steps in the interval, was also undetectable even at n = 448, so
this interval has little influence on the final distribution at these sample sizes. FID parity
here therefore says little about transition accuracy, which is better measured by item 2.

**4. Does GMMD reduce computation once fitting is included?** No; it increases it. The segment
goes from 40 to 1,200 score evaluations (30×) and from 1.46 to 43.75 s per image. A full image
goes from 2,000 to 3,160 NFE and from 73.2 to 116.0 s (+58%), and throughput drops from 49 to
31 images per GPU-hour. Fitting happens once per image and online; nothing can be amortised
offline. The cost is almost all score evaluations (36.5 ms per NFE, the same rate as the teacher).

**5. Better quality–cost tradeoff than the coarse teacher?** No. The coarse teacher costs 4 NFE
(0.15 s) for the segment against GMMD's 1,200 NFE (43.75 s), with indistinguishable FID (58.0 vs
58.1). Costs could not be matched: GMMD costs more than even the full 40-step original segment,
so no "coarse" setting can reach it from below. On this interval the coarse teacher dominates.
Neither changes total cost much, because the interval is only 2% of the sampler (coarse saves
1.3% per image).

**6. Remaining limitations.**
- **Sample size.** 192 images per method; FID is dominated by small-n bias and P/R is noisy. The
  planned 1,024-image run was stopped by decision. All rankings are preliminary.
- **Candidate set.** Only two intervals, both starting at t = 0.60. A short interval elsewhere
  might be approximated better.
- **GMMD family and settings.** K = 1, isotropic, uniform weights, prior-free init. 300 iterations
  was the largest count tested, and error was still falling steeply (1.32 → 0.59 → 0.04). The
  mean gradient fell only to ~2% of its starting value, so the fit is likely under-converged.
  A plausible saving, untested: for K = 1 the initial mean need not be perturbed by √gap·z
  (perturbation only separates components when K > 1), which may account for many of the
  iterations.
- **Target mismatch.** IBW fits the continuous-time conditional; the teacher segment is the
  discretised 40-step kernel. Some of the residual error may be this gap, not fit quality.
- **Validation metric.** DINOv2 on Tweedie-denoised states with n = 64 per set; detection rates
  are relative to that power.
- **Timing.** Costs are synchronised per-stage measurements summed across stages, with the prefix
  computed once and reused across methods. The uncached end-to-end x_T → x_0 timing was planned
  but not run. The GPUs were shared, so absolute seconds carry some noise; NFE counts do not.
- **Numerics.** Sampling and fitting were full fp32. FID/P-R feature extraction used PyTorch's
  default TF32 convolutions, identically for the reference and all methods.
- **Environment.** torch went from 2.12.0 to 2.12.1 (same CUDA 13.0 build) when torch-fidelity
  was installed; the checkpoint/vendor/IBW parity tests pass. The exp4 code is uncommitted on
  top of `985d848`.

## 6. Cost of this experiment and how to extend it

| phase | wall-clock | GPU-hours |
|---|---|---|
| Validation (16 anchors, 2 GPUs) | 2.7 h | ≈ 5.4 |
| Pilot: shared prefix 1.57, teacher 2.33, coarse 2.28, GMMD 4.61 GPU-h | 7.3 h | 10.8 |
| Stopped 1,024 run (teacher complete, coarse 7/16 chunks) | 11.7 h | ≈ 23 |

```bash
run/exp4_validation.sh                                   # stage 1 (resumable; selection.json is frozen once written)
N_CHUNKS=3  TAG=pilot run/exp4_generate.sh               # pilot (already done; reruns skip finished chunks)
N_CHUNKS=16 TAG=n1024 E2E=1 run/exp4_generate.sh         # resume the 1,024 run: ≈ 31 GPU-h (~16 h on 2 GPUs) remaining
CUDA_VISIBLE_DEVICES=0 python run/exp4_supplementary_fid.py
# unattended:  tmux new-session -d -s exp4 'bash run/exp4_all.sh 2>&1 | tee -a run/logs/exp4_all.log'
```

**Artifacts** (under `outputs/exp4_interval_replacement/`):
- `validation/`: per-anchor JSON and x_t, `candidates.{md,json}`, frozen `selection.json`
- `generation/{prefix,teacher,coarse,gmmd}/chunk*.{npz,json}`: images, x_s, seeds, per-stage
  time / NFE / peak memory
- `metrics_pilot.json`, `results_table_pilot.md`, `metrics_supplementary.json`, `figures/`, `STATUS`

Code: `gmmd/pipeline.py`, `gmmd/ibw_batched.py`, `gmmd/fid.py`,
`gmmd/experiments/interval_{validation,replacement}.py`, `configs/exp4_interval_replacement.toml`,
`tests/test_interval_replacement.py`, `run/exp4_*.sh`.
