"""exp5: an amortized IBW-GMMD sampler -- ten conditional 8-component mixture draws in place of
the complete 2,000-step reverse-diffusion trajectory.

Modules:

* :mod:`.grid`        -- the 10-interval partition of the teacher grid, teacher-side rollouts,
  the terminal noise-removal map, and the coarse teacher baselines;
* :mod:`.ibw_targets` -- offline stage A: teacher-free initialisations and batched IBW fits of
  the per-anchor conditional, with convergence diagnostics and a resumable cache;
* :mod:`.student`     -- stage B: the conditional mixture network and its likelihood;
* :mod:`.sampler`     -- the standalone 10-transition sampler (student checkpoint only);
* :mod:`.metrics`     -- conditional-fidelity metrics shared by stages A, B and C.

The experiment driver is :mod:`gmmd.experiments.ibw_amortized`.
"""
