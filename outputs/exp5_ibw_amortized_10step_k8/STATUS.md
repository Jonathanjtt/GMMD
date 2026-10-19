# exp5 status (amortized IBW–GMMD, 10 transitions, K = 8)

Branch `exp5-ibw-amortized` (from `a03cf0a`). Commit locally at milestones; never push.
Untracked `results/conditional_multimodality/ibw_K40_B10*` belong to earlier work — do not touch.

## User decisions (2026-10-19)
- Precision: TF32 for offline IBW fitting, terminal pushforward bank, student training/inference.
  fp32 (TF32 off) for every teacher reference: anchor trajectories, reference rollouts, coarse
  baselines, ablation endpoints, evaluation denoising/features. Principal-direction init runs fp32.
- Budget: ~1.5 days on GPUs 0+1 after the pilot (64 train anchors/interval + refinement + ablation).
  GPUs 2/3 are other users'. GPUs 0/1 also host another user's light job (~2.4 GB).
- IBW init: K = 8, teacher-sample-free, components started far apart; initial mean AND covariance
  are tunable on validation anchors (variants in the config; no teacher-endpoint init).
- Session handoff: the conversation continues in the Claude CLI inside tmux `exp5-claude`.

## Measured (H200, NCSN++ 65.6 M params)
fp32 37 ms/img/NFE, TF32 11 ms, bf16 10 ms (batch 64–128); training fwd+bwd 108 / 36 / 29 ms/img.

## Done
- [x] Code: `gmmd/exp5/` (grid, ibw_targets, student, sampler, metrics, train, evaluation, report, figures),
      driver `gmmd/experiments/ibw_amortized.py`, tests `tests/test_exp5.py` (13 CPU + 1 GPU, all pass;
      existing fast suite 78/78 pass). Commits c2e13fb, 083cd64 (local only).
- [x] DESIGN.md: section-1 record, partition table, terminal handling, step-size scaling, rules.
- [x] Partition + split manifests; val anchors (4 trajectories).
- [x] P0 step-size check (prior-free K=8, val 0-1, intervals 1/4/7): c = 0.5 chosen (convergence by
      50-75 it; same converged fidelity as 0.2/1.0), budget 300 it. Figure `figures/fig_p0_ibw_step_size`.
      Finding: converged isotropic IBW under-disperses content directions -> large DINOv2 excess
      (0.15-1.0) while pixel excess is ~0 at high noise; early transients score better ("over-contraction").
- [x] Principal-direction init checked on the real teacher: top Rayleigh quotients 1.0-2.0 x gap
      (multimodal/broad posterior directions), eigenvectors not converged (overlap 0.57-0.84), ~108 fp32 NFE.

## Running (tmux)
- `exp5-gpu`: run/exp5_pilot_p0.sh — GPU0 val reference rollouts (40 pairs, ~13:25 UTC);
  GPU1 test + train anchor trajectories.
- `exp5-p1`: run/exp5_pilot_p1.sh — P1: 5 init variants x 40 val pairs (GPU1 now, GPU0 after refs),
  stage-A evalfits, then the pre-registered selection -> selection.json (~15:20 UTC).

## Next
- [ ] P2 (run/exp5_pilot_p2.sh): freeze val, pilot train fits (trajs 0-7), pilot student (3k steps),
      stage A/B/C on val, 192-image rollouts + FID (preliminary), rollout shift; GPU1 starts main fits.
- [ ] Main: 64 train anchors, 8 test anchors (refs + eval fits), student, stage A/B/C, 1,024 images,
      FID, timing, rollout-shift refinement, direct-NLL ablation, REPORT.md
