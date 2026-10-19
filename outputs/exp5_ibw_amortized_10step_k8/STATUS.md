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

## Done (2026-10-19, main run so far)
- [x] P2 pilot (8 trajs, 3,000 steps): stage B at the IBW replicate floor j0-j8; end to end the trained
      pilot is WORSE than its untrained init (FID n=192: 126.2 vs 98.3); IBW targets carry a mean bias
      (the Tweedie Gaussian is closer to the teacher per interval). Commit f749bc1.
- [x] Main fits: train 640 (64 trajs), test 80 + replicate frozen_r1 80. Main student (12,000 steps,
      val NLL 1.4084, flat from ~4,500; ckpt_best = step 4,750). Commits 73ce166, c9f418c.
- [x] IBW chain (online IBW, 64 images, 8,001 teacher evals/img): washed out like the student.
- [x] FID n = 1,024 (paired bootstrap): teacher 22.55, untrained init 77.61, ANC10 94.38,
      **main 111.73**, RD10 442.16 (`fid_n1024_first.json`, `fid_n1024_init.json`); n = 64 incl. the IBW
      chain in `fid_n64_ibwchain.json`.
- [x] Detail diagnostic (`gmmd/exp5/diag_detail.py`, val): IBW targets keep 0.24-0.65 of the teacher's
      fine detail at j1-j7, the Tweedie Gaussian 0.87-1.04; the student copies the targets.
- [x] Rollout shift (main): pixel dispersion of boundary states 0.98 (b1) -> 0.59 (b6) -> 0.51 (b9) of
      the teacher's (pilot 0.73 at b9) -> refinement is warranted.
- [x] Refinement (22:15 UTC): 32 student-rollout trajectories, 288 frozen IBW fits (j1-j9, all finite,
      none stopped), fine-tuned 4,000 steps from main ckpt_best (val NLL 1.4084 -> 1.4080; only j9
      moves). Shift: pixel dispersion at b5/b6/b9 0.76/0.59/0.51 -> 0.99/0.90/0.78 (floor 1.03-1.08),
      pixel excess at b9 0.038 -> 0.021. `eval/shift_student_refined.json`, `figures/fig_training_refined`,
      `figures/fig_trajectory_student_refined`. BUT FID n = 1,024 gets worse: refined 117.46 vs main
      111.73, paired CI [+4.2, +8.2]; contrast unchanged 0.124 (`fid_n1024_refined.json`).

## Running (tmux; check `tail outputs/exp5_ibw_amortized_10step_k8/RUNLOG`)
- `exp5-main0b` (GPU0): `run/exp5_main_part2.sh` -- (refinement: done) -> ablation (train endpoints E = 4,
  ~5.3 h from 22:15, student `ablation_endpoints` ~1.8 h, 1,024 images) -> FID n1024_students -> test refs (shared) ->
  evalstudent on test (main, refined, ablation_endpoints, init_main) -> detail on test -> timing.
- `exp5-main1` (GPU1, shared with another user): `ONLY=gpu1 bash run/exp5_main.sh` -- test refs
  (80 pairs, fp32) -> endpoints (shared claims) -> ANC20/50 1,024 images -> (IBW chain: done).
- If a session dies: rerun the same script; every stage skips finished units and reclaims stale claims.

## Next
- [ ] After both chains: `$M fid --tag n1024 --n 1024 --methods student:main student:refined
      student:ablation_endpoints student:init_main coarse:reverse_diffusion:10 coarse:ancestral:10
      coarse:ancestral:20 coarse:ancestral:50` (one file, one bootstrap), figures
      (`python -m gmmd.exp5.figures`; stage ABC + kernels vs teacher on test; quality vs NFE),
      `$M tables --abc main refined ablation_endpoints init_main --fid n1024`, `$M costs --student main` (break-even of the deployed main student; refinement data is its own category),
      REPORT.md short answer + sections 4-8, commit locally.
