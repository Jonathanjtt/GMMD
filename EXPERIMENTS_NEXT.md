# Three focused follow-up experiments

All jobs use the project environment at `/scratch/jt4994/GMMD/.venv`. GPU jobs follow this
account's established successful convention, `torch_pr_468_general` on `l40s_public`, and request
one 48 GB L40S. No script uses `srun`. Experiment 2 is CPU-only and uses the confirmed
`cpu_short` partition. Each script activates the
environment, prepares its own runtime/cache directories, and is reproducible from one submission.

## 1. Transition length and conditional multimodality

**Question.** Across several fixed noisy states, does evidence for conditional multimodality tend
to strengthen as the reverse transition becomes longer?

- Submit: `sbatch run/exp1_transition_multimodality.sbatch`
- Entry point: `python -m gmmd.experiments.transition_multimodality_sweep --config exp1_transition_multimodality`
- Config: `configs/exp1_transition_multimodality.toml`
- Output: `outputs/exp1_transition_multimodality/`
- Images: validation indices `0, 137, 1024`
- Forward-noise seeds: `3101, 3102, 3103`
- Fresh rollout seeds: `21000–21063`, `22000–22063`, `23000–23063`
- Paired endpoints: `0.60→{0.58, 0.50, 0.45, 0.35}`; 64 rollouts each
- Prespecified representation: DINOv2 ViT-B/14; existing 10-D PCA, diagonal GMM/BIC/CV,
  bootstrap stability, and Gaussian-null calibration

Main figures are `figure_1a_multimodality_vs_length`, `figure_1b_preferred_k_heatmap`, and
`figure_1c_representative_embeddings`, each saved as PNG and PDF. Raw samples, DINOv2 features,
PCA scores, cluster labels, CSV/JSON summaries, resolved config, provenance, and `SUMMARY.md` are
stored in separate subdirectories. Endpoints share the same stochastic paths within each fixed
`x_t`, so the comparison is paired.

Regenerate aggregate plots without rerunning the teacher:

```bash
python -m gmmd.experiments.transition_multimodality_sweep \
  --config exp1_transition_multimodality --plot-only
```

## 2. Controlled toy H1 test

**Question.** Does the approximation benefit of a two-component mixture increase as an exact 2D
target changes from unimodal to strongly separated?

- Submit: `sbatch run/exp2_toy_h1.sbatch`
- Entry point: `python -m gmmd.experiments.toy_h1 --config exp2_toy_h1`
- Config: `configs/exp2_toy_h1.toml`
- Output: `outputs/exp2_toy_h1/`
- Separations: `0, 1, 2, 4, 6`; seeds `0–4`
- Target components: equal weight, identity covariance, means `(-d/2,0)` and `(d/2,0)`
- Fits: sample direct-fit full-covariance K=1/K=2 reference, plus existing uniform-isotropic
  IBW K=1/K=2. These are reported separately rather than treated as the same family.
- Metrics: held-out NLL, sliced Wasserstein, MMD², and nearest-halfspace mode-mass error

Main figures are `figure_2a_gmm_advantage`, `figure_2b_distribution_snapshots`, and
`figure_2c_error_vs_separation`. CSV/JSON results include per-seed values and signs are defined so
positive advantage means K=2 is better. IBW parameter/KL trajectories are saved in NPZ form.

Regenerate plots:

```bash
python -m gmmd.experiments.toy_h1 --config exp2_toy_h1 --plot-only
```

## 3. IBW mode-discovery / initialisation diagnostic

**Question.** On the canonical `0.60→0.45` H1 conditional, is poor fitting primarily insufficient
K or failure to discover separated modes?

- Submit: `sbatch run/exp3_ibw_initialization.sbatch`
- Entry point: `python -m gmmd.experiments.ibw_initialization --config exp3_ibw_initialization`
- Config: `configs/exp3_ibw_initialization.toml`
- Output: `outputs/exp3_ibw_initialization/`
- Optimisation seeds: `11, 29, 47`
- Conditions: K=1 standard; K=2 standard; K=4 standard; K=2 deliberately spread without teacher
  labels; K=2 DINOv2 teacher-cluster oracle
- Fit settings: IBW, 300 iterations, step size 0.1, four gradient samples
- Metrics: DINOv2 MMD², DINOv2 sliced Wasserstein, raw pairwise center distances, separation
  divided by `sigma_i+sigma_j`, collapse flag, and effective separated-component count

Because a fresh clone omits the gitignored H1 arrays, this job reproducibly rebuilds the canonical
H1 run under `outputs/exp3_ibw_initialization/reference_h1/`; it never modifies old `results/`.
Per-fit JSON/NPZ checkpoints permit a requeued job to reuse completed fits. A fit is called
collapsed only when at least one normalized pairwise separation is below 1.0; raw values remain
available and the binary flag is not used alone.

Main figures are `figure_3a_fit_quality`, `figure_3b_component_separation`,
`figure_3c_quality_vs_separation`, and `figure_3d_component_evolution`.

Regenerate plots:

```bash
python -m gmmd.experiments.ibw_initialization \
  --config exp3_ibw_initialization --plot-only
```

## Completed continuation

The compute-conscious continuation preserves every result above and adds two independent output
trees.

- Expanded Experiment 1: `configs/exp1_transition_multimodality_expanded.toml`, five fresh fixed
  states with 128 paired rollouts per endpoint, output under
  `outputs/exp1_transition_multimodality_expanded/`.
- Combined Experiment 1 analysis: eight states across both cohorts, generated without rerunning
  the teacher by `python -m gmmd.experiments.transition_multimodality_combined`, output under
  `outputs/exp1_transition_multimodality_combined/`.
- Experiment 3 capacity/optimisation: `configs/exp3_ibw_capacity.toml`, a resumable K/BG/init/
  iteration/step-size sweep with 128-sample held-out evaluation and a 256-sample independent
  teacher pool, output under `outputs/exp3_ibw_capacity/`. The final focused stage maps
  50–1000-step K=8 trajectories at step 0.01 and repeats the selected 400-step point across
  seeds 11, 29 and 47; completed checkpoints are reused on every submission.

Submit the GPU continuations with:

```bash
sbatch run/exp1_transition_multimodality_expanded.sbatch
sbatch run/exp3_ibw_capacity.sbatch
```

Regenerate Experiment 3 figures and its generated summary without fitting or sampling:

```bash
python -m gmmd.experiments.ibw_capacity_sweep \
  --config exp3_ibw_capacity --plot-only
```

## Shared execution details

Experiments 1 and 3 independently verify/download the checkpoint, validation shard, and DINOv2
weights. A shared `flock` serialises only this one-time asset preparation if the two jobs start
together. Resolved settings, package/GPU provenance, seeds, raw numerical outputs, report-friendly
PNG/PDF figures, and a generated `SUMMARY.md` live inside each experiment's output directory.

Interpretation guardrails retained from the current repository:

1. Experiment 1 measures multimodality in a prespecified semantic representation; it does not
   claim a universally monotone model order.
2. Experiment 2 distinguishes direct-fit capacity from IBW optimisation.
3. Experiment 3 optimises the existing continuous-time VE Bayes conditional score, not the
   intractable density of the discretised reverse-SDE transition; teacher initialisation is an
   oracle diagnostic only.
