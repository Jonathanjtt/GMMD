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
- [x] Read code/reports; recorded what IBW targets (design notes to be written to DESIGN.md)
- [x] `gmmd/exp5/grid.py` partition, trajectories, interval rollouts, terminal denoise, coarse baselines
- [x] `gmmd/exp5/ibw_targets.py` inits (prior_free / principal / random_far), fit wrapper, mixture utils
- [x] `gmmd/ibw_batched.py`: diag_every diagnostics + ITERATION_COUNTER (defaults unchanged)
- [x] `gmmd/teachers/score_sde/teacher.py`: process-wide SCORE_EXAMPLES / INSTANCES counters
- [x] `gmmd/exp5/student.py` NCSN++-initialised mixture student + likelihood
- [x] `gmmd/exp5/sampler.py` standalone instrumented 10-transition sampler
- [x] `gmmd/exp5/common.py`, `gmmd/exp5/metrics.py`, `configs/exp5_ibw_amortized_10step_k8.toml`

## Next
- [ ] driver `gmmd/experiments/ibw_amortized.py` (partition, anchors, refs, fit, evalfits, select, bank)
- [ ] `gmmd/exp5/train.py` (amortization, resume, EMA, val NLL) and evaluation stages
- [ ] `tests/test_exp5.py` on the CPU toy teacher; run existing test suite
- [ ] DESIGN.md (section-1 record, partition table, terminal handling, step-size scaling, rules)
- [ ] Pilot: anchors (all splits) → P0 step-size check → P1 init variants on 4 val anchors →
      frozen selection → P2 8 train anchors + pilot student + pilot rollouts/FID(192)
- [ ] Main: 64 train anchors, 8 test anchors (refs + eval fits), student, stage A/B/C, 1,024 images,
      FID, timing, rollout-shift refinement, direct-NLL ablation, REPORT.md
