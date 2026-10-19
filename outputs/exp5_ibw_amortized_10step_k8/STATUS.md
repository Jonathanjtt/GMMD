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

## Done (2026-10-19, continued)
- [x] P1: 5 teacher-free inits x 40 val pairs x 300 it; stage-A evaluation at 25/50/100/200/300.
      Pre-registered selection frozen 15:47 UTC: **prior_free @ 25 iterations, c = 0.5**
      (`selection.json`; every other init +0.16..+0.19 DINOv2 excess, CIs exclude 0). All inits reach
      the same fixed point by 300 it; the selected snapshot is an early-stopped transient (NOTES.md).
      Figure `figures/fig_p1_stage_a_vs_iterations`. Commit dd2534a.
- [x] Fixes: P1 evalfits race (`run/exp5_pilot_p1.sh` waits for the fit lock; `run/exp5_pilot_p1_finish.sh`).
- [x] Added (post-hoc, labelled): `frozen_r1` replicate fits (stage-B IBW-vs-IBW floor), IBW-chain
      diagnostic (`gmmd/exp5/oracle.py`, `generate --method ibwchain`), `prior_free_fine` iteration grid,
      claims for generation chunks and evaluation pairs (both GPUs can share them).

## Running (tmux)
- `exp5-p2`: run/exp5_pilot_p2.sh — pilot fits done (trajs 0-7); pilot student (3,000 steps, GPU0),
  then stage A/B/C on val, 192-image rollouts (student, RD10, ANC10), preliminary FID, shift.
  Background: main train fits trajs 8-63 on GPU1 (`run/logs/exp5_main_fits_gpu1.log`).
- `exp5-fine`: post-hoc fine iteration grid on val (fit + evalfits).  `exp5-rep`: val frozen_r1 fits.
- GPU1 shared with another user's jobs since ~15:40 UTC (3-5x slower); GPU0 carries the critical path.

## Next
- [ ] Pilot gate (val NLL falls, covariances valid, components used, 10-transition generation works,
      FID vs RD10/ANC10 at n = 192, shift) -> `tmux new-session -d -s exp5-main 'bash run/exp5_main.sh'`
      (v2: shared claims; ablation endpoints E = 4 = the IBW targets' 800 score evaluations per anchor).
- [ ] Refinement if the shift is substantial (`run/exp5_refine.sh`); timing (`run/exp5_timing.sh`,
      fp32 and TF32 on both sides, GPU0 quiet); FID n = 1,024 for all; figures; costs; REPORT.md.
