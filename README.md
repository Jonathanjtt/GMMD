# GMMD

**Is the stochastic transition of a diffusion model genuinely multimodal?**

A pretrained score-based diffusion model defines a stochastic map from a noisy image `x_t` at
diffusion time `t` to a less noisy one `x_s` at an earlier time `s < t`. Running the reverse
SDE many times from the *same* `x_t` gives many different `x_s`. The question this repository
asks is whether the resulting conditional

$$p_\theta(x_s \mid x_t)$$

has several distinct modes, or one broad one — and, if it is multimodal, whether a **mixture of
isotropic Gaussians** fitted by **IBW** approximates it better than a single Gaussian.

This is an early-stage research repository. Right now it contains the infrastructure and the
**first diagnostic experiment (H1)**: characterise `p_θ(x_s | x_t)` for one fixed `x_t`. The
variational half is deliberately not wired up yet — see [Status](#status).

---

## The question, and the trap

CelebA-HQ is obviously multimodal: it contains many different faces. That is not interesting and
not what is being asked. The claim under test is about the **conditional**: having fixed one
noisy tensor `x_t`, does the teacher's own stochasticity still carry the sample to genuinely
different outcomes?

So the experiment is built around one discipline: **`x_t` is drawn once and frozen.** Every
rollout starts from bit-identical bytes (its SHA-256 is recorded), and only the reverse-SDE
noise differs between them. Two controls keep the measurement honest:

| Control | What it rules out |
| --- | --- |
| The probability-flow **ODE** from the same `x_t`, run repeatedly | that the spread is numerical noise or a bug in the fixed-`x_t` plumbing — repeats must be identical |
| A **short** transition `t → s_near` instead of `t → s` | that any long-transition spread is trivial; a short transition should stay local |

---

## What is in here

| Module | What it does |
| --- | --- |
| [`gmmd/teachers/score_sde/`](gmmd/teachers/score_sde/) | the stochastic teacher: Yang Song et al.'s official score-SDE implementation, **vendored** (see [Attribution](#attribution)), wrapped so a single frozen `x_t` can be rolled out many times with independent seeds |
| [`gmmd/datasets/celebahq.py`](gmmd/datasets/celebahq.py) | CelebA-HQ 256×256 with *exactly* the checkpoint's preprocessing |
| [`gmmd/diagnostics.py`](gmmd/diagnostics.py) | multimodality **diagnostics** for a sample cloud — model selection, a bootstrap likelihood-ratio test, clustering stability against a unimodal null |
| [`gmmd/ibw.py`](gmmd/ibw.py) | Petit-Talamon, Lambert & Korba's Algorithm 1 (IBW / MD / NGD) for uniform-weight isotropic Gaussian mixtures |
| [`gmmd/vi.py`](gmmd/vi.py) | targets for that optimiser — **and a written record of why the teacher conditional is not yet one of them** |
| [`gmmd/experiments/`](gmmd/experiments/) | `conditional_multimodality`, the H1 experiment |

### The teacher

[`ScoreSDETeacher`](gmmd/teachers/score_sde/teacher.py) exposes one operation:

```python
transition = teacher.sample_transition(x_t, t=0.60, s=0.45, seeds=range(64))
#  -> 64 independent x_s, all from that one x_t, 299 network evaluations each
```

It runs the **reverse SDE**, not the probability-flow ODE and not DDIM — a deterministic sampler
would make the conditional a point mass and the question vacuous. The loop is upstream's own
predictor–corrector loop restricted to the index range `[i_t, i_s)` of the official 2000-point
time grid; each rollout draws its noise from its own generator, so rollout *i* is reproducible
from its seed alone. `teacher.ode_transition(...)` is the deterministic control on the same grid.

**Time convention** is Song et al.'s, unchanged: continuous `t ∈ [ε, 1]` with `ε = 1e-5`, and for
the VE SDE `σ(t) = σ_min (σ_max/σ_min)^t` with `σ_min = 0.01`, `σ_max = 348`. So `t = 0.60` is
`σ ≈ 5.3` and `s = 0.45` is `σ ≈ 1.1`. Requested times snap to the official grid and the
realised values are what gets recorded.

### The checkpoint, and what it is

The only CelebA-family checkpoint the upstream repository offers is **CelebA-HQ 256px, NCSN++,
VE SDE** (`ve/celebahq_256_ncsnpp_continuous/checkpoint_48.pth`). Upstream states that the
original CelebA and CelebA-HQ checkpoints could not be released and that this is a model
retrained with personal resources; there is **no** 64×64 CelebA checkpoint. Two consequences are
recorded rather than glossed over:

- The teacher was trained on **all 30 000** CelebA-HQ images. No image is held out from *its*
  training; "held-out" here means held out of this experiment's fitting, not of the teacher's.
- The images come from a Hugging Face mirror of CelebA-HQ at 256×256. How that mirror produced
  its 256×256 PNGs from the 1024×1024 originals is undocumented, so it is not guaranteed to be
  bit-identical to upstream's own `tfrecords`. Any difference is far below the `σ ≥ 0.01` noise
  the teacher always sees, but it is not nothing.

Preprocessing matches the checkpoint exactly: uint8 → float32 in **[0, 1]** (*not* [-1, 1] — the
network centres its own input), CHW, RGB, no resizing. `tests/test_dataset.py` pins this, and
`tests/test_checkpoint.py` checks compatibility end to end by denoising a lightly-noised dataset
image and requiring > 30 dB PSNR, with the wrong value range as a negative control.

### The diagnostics

Nothing here *proves* multimodality, and the code says so in as many words. Each number answers
a narrower question, and all of them are reported side by side for the long transition, the short
transition and the deterministic control:

- **pairwise distances** in pixel and feature space — spread, which a single broad mode has too;
- **Gaussian mixtures with K = 1, 2, …** in a PCA subspace, scored by BIC and by held-out
  log-likelihood under K-fold cross-validation;
- **a parametric-bootstrap likelihood-ratio test** of K = 1 against K = 2 (the classical
  asymptotics do not apply to mixtures, so the null is simulated from the fitted single
  Gaussian);
- **clustering stability**: bootstrap adjusted Rand index and silhouette of a 2-means split,
  each **calibrated against a unimodal Gaussian null** with the cloud's own covariance — because
  k-means always returns two clusters and a silhouette of 0.3 means nothing on its own;
- **PCA and UMAP** views, labelled as views. A UMAP picture that separates points is not
  evidence.

The stated assumptions: the feature space is treated as Euclidean and the null is Gaussian in the
PCA subspace, so an elongated but unimodal *non-Gaussian* cloud can reject it. The sample grids
are the guard against reading "non-Gaussian" as "multimodal".

---

## Install

```bash
git clone git@github.com:<you>/GMMD.git && cd GMMD
pip install -r requirements.txt     # install torch first, per https://pytorch.org
pip install -e .                    # the package itself declares no dependencies (see pyproject.toml)
```

Python ≥ 3.11 (the config loader uses the stdlib `tomllib`). A GPU is needed for anything
involving the teacher: one 256×256 network evaluation is ~25 ms on an H200, and a 299-step
rollout is ~300 of them.

### Data

```bash
python -m gmmd.download --checkpoint ve/celebahq_256_ncsnpp_continuous --celebahq validation
```

This fetches ~1.2 GB into `data/` (gitignored) and verifies the SHA-256 of each file. The
checkpoint comes from the upstream Google Drive folder and needs `pip install gdown`. Set
`$GMMD_DATA_DIR` to put the data somewhere else.

---

## Run the experiment

```bash
CUDA_VISIBLE_DEVICES=<free gpu> python -m gmmd.experiments.conditional_multimodality \
    conditional_multimodality
```

Everything is driven by [`configs/conditional_multimodality.toml`](configs/conditional_multimodality.toml)
— dataset and image index, checkpoint, SDE and sampler, discretisation steps, `t`, `s`, `s_near`,
the number of rollouts, every seed, batch size, feature extractor, the diagnostic budgets, and
the output directory. No machine-specific path is hardcoded. Any key can be overridden on the
command line, and the resolved config is written next to the results:

```bash
... conditional_multimodality --set transition.n_samples=8 --set features.extractor=dinov2_vitb14
```

A variant file declares `inherit = "<base>"` and states only what it changes;
[`configs/conditional_multimodality_pc.toml`](configs/conditional_multimodality_pc.toml) is the
same experiment under the official predictor–corrector sampler.

### Which sampler is "the teacher"?

Two are available on the same time grid, and they are **different transition kernels**, not two
qualities of one:

- `predictor = "reverse_diffusion"`, `corrector = "none"` — the exact SMLD discretisation of the
  reverse SDE. This is the base config.
- `corrector = "langevin"` — upstream's default predictor–corrector sampler for this checkpoint.
  Its Langevin steps re-target the *marginal* at every noise level, so the induced conditional is
  not the reverse SDE's own.

Both are run and compared rather than one being declared correct.

### Output

Each run writes a directory under `results/`:

| File | Contents |
| --- | --- |
| `sample_grid.png` | `x₀`, the frozen `x_t`, and every `x_s⁽ⁱ⁾` — raw, and through the teacher's own denoiser, because a raw `x_s` at `σ ≈ 1` is unreadable |
| `controls.png` | the ODE repeats and the short transition |
| `embedding_pca.png`, `embedding_umap.png` | the three groups in one joint 2-D view |
| `gmm_selection.png`, `lrt_null.png` | BIC and held-out likelihood against K; the bootstrap LRT null with the observed statistic |
| `diagnostics.json` | every diagnostic number, with its settings |
| `provenance.json` | git commit, full config, package versions, GPU, checkpoint SHA-256, dataset, image identifier, the SHA-256 of `x_t`, every seed, realised `t`/`s`, NFE, runtimes |
| `samples.npz` | the raw arrays (gitignored; `provenance.json` has everything needed to regenerate them) |

---

## Status

**Working and tested.** The teacher and its fixed-`x_t` transition sampler; CelebA-HQ loading;
the diagnostics; the IBW optimiser; the H1 experiment end to end.

**Blocked, on purpose.** IBW is *not* run against the teacher conditional. Reverse-KL variational
inference needs the target's score, `∇_{x_s} log p_θ(x_s | x_t)`. The teacher provides the
**marginal** score `∇ log p_s(x_s)` and a *sampler*; the transition kernel is a composition of
hundreds of Gaussian steps and has no closed-form density or score. Substituting the marginal
score, a maximum-likelihood fit to teacher samples, or the continuous-time Bayes-rule surrogate
would each answer a different question. `gmmd.vi.TeacherConditionalTarget` therefore raises, with
the candidate routes written out, rather than quietly picking one. See the module docstring of
[`gmmd/vi.py`](gmmd/vi.py).

**Not started.** Multiple `(t, s)` pairs, multiple `x_t`, the single-Gaussian baseline, the
quality-versus-NFE comparison. The layout anticipates them; none is implemented.

---

## Tests

```bash
pytest                       # fast; CPU only; no checkpoint, no network
pytest -m network            # also checks the vendored and reference code against upstream
pytest -m checkpoint         # the real 1 GB checkpoint on a GPU
```

The fast suite runs the teacher against an **analytic Gaussian score** in two dimensions, where
the reverse-diffusion sampler is an affine Gaussian recursion whose transition kernel can be
propagated in closed form — so the sampler, the seeding, the fixed-`x_t` semantics and the
deterministic control are checked exactly rather than by eye. It also verifies that the
discretised kernel converges to the exact time-reversal posterior as the grid is refined.

---

## Attribution

**The teacher** is [`yang-song/score_sde_pytorch`](https://github.com/yang-song/score_sde_pytorch)
(Apache-2.0), from *Score-Based Generative Modeling through Stochastic Differential Equations*,
Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole, ICLR 2021. Its sources are vendored under
[`gmmd/teachers/score_sde/_vendor/`](gmmd/teachers/score_sde/_vendor/) with the licence intact.
`_vendor/provenance.py` pins the upstream commit, records the SHA-256 of every file, and declares
the complete list of modifications — relative imports, a noise hook so per-rollout generators can
be installed without touching the update rules, and the pure-PyTorch `upfirdn2d` path (no CUDA
extension is built). `tests/test_vendor_parity.py` re-downloads upstream and proves the vendored
copy is that source plus exactly those edits.

**IBW** implements *Variational Inference with Mixtures of Isotropic Gaussians*, Marguerite
Petit-Talamon, Marc Lambert & Anna Korba, NeurIPS 2025 ([arXiv:2506.13613](https://arxiv.org/abs/2506.13613)).
[`gmmd/ibw.py`](gmmd/ibw.py) is written from the paper's published equations — Algorithm 1,
Eq. (12), (13), (14) and Proposition 3.1 — and **not** copied from the authors' reference
implementation at [margueritetalamon/VI-MIG](https://github.com/margueritetalamon/VI-MIG), which
carries no licence file. `tests/test_ibw_reference_parity.py` fetches that reference at a pinned
commit *at test time* and asserts the two agree to machine precision on the score, the
Proposition 3.1 gradients and all four update formulas.

**Data**: CelebA-HQ, from *Progressive Growing of GANs*, Karras, Aila, Laine & Lehtinen, 2018,
itself derived from CelebA (Liu, Luo, Wang & Tang, 2015). Check those datasets' terms before use.

## Licence

Not yet chosen — until one is added, default copyright applies and no reuse is granted.
Apache-2.0 would be the natural choice, since it is what the vendored teacher code carries.
