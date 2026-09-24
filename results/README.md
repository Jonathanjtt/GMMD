# results/

One directory per run, named by the config's `output.dir`. Figures, `diagnostics.json` and
`provenance.json` are tracked; the raw `samples.npz` arrays are not (see `.gitignore`), because
`provenance.json` records every seed needed to regenerate them bit-for-bit.

| Directory | What it is |
| --- | --- |
| `smoke_n8/` | A **smoke run**: only 8 conditional rollouts, reduced bootstrap budgets. It exists to show the output format and to prove the pipeline runs end to end. **It is not a result.** Eight points cannot support any claim about multimodality, and its own diagnostics say so — every model-selection and bootstrap number in it is consistent with a single mode at that sample size. |

Reproduce a directory with the command in its `provenance.json` (`argv`), at the git commit
recorded there.
