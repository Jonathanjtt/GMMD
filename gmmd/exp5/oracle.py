"""IBW chain: online IBW in place of the network at every one of the ten transitions.

A DIAGNOSTIC, never the sampler.  From the same ``x_T`` as every other method, each interval
runs the frozen IBW configuration (initialisation, step size, iteration count) at the CURRENT
state and makes one draw from the fitted mixture (uniform weights, as IBW fits them); the
terminal draw goes through the teacher's noise removal, exactly like the student's cached
bank.  The result is what a perfect amortisation of the IBW targets -- one with no
approximation error and no rollout shift -- would generate, so it separates the quality of the
targets (stage A, end to end) from the amortisation (stage B).  Cost: 10 x iterations x K x B
score evaluations per image (8,000 at the frozen 25 x 8 x 4), TF32 like the offline fits.
"""

from __future__ import annotations

import numpy as np
import torch

from gmmd import pipeline as P
from gmmd.exp5 import common as C
from gmmd.exp5 import grid as G
from gmmd.exp5 import ibw_targets as IT

NS_CHAIN_INIT, NS_CHAIN_NOISE, NS_CHAIN_DRAW = 92, 93, 94


def ibw_chain(cfg, teacher, seeds, variant="frozen", batch_size=64):
    """Images ``(n, C, H, W)`` (device) and a record, for image seeds ``seeds``."""
    import gmmd.experiments.ibw_amortized as D
    parts = C.partition(cfg, teacher)
    vid, v, step_c, iters, _ = D._variant_settings(cfg, variant, "oracle")
    ci = cfg["ibw"]
    K, Bg = int(ci["n_components"]), int(ci["n_grad_samples"])
    seeds = [int(s) for s in seeds]
    x = P.prior_sample(teacher, seeds).to(teacher.device)                  # the teacher's own x_T
    nfe, per_interval = 0, []
    for iv in parts:
        m0, v0, info = IT.initial_mixture(v["scheme"], teacher, iv, x, K,
                                          [G.seed_of(NS_CHAIN_INIT, s, iv.j) for s in seeds],
                                          spread=float(v["spread"]), var=v["var"], batch_size=batch_size)
        res = IT.fit_interval(teacher, iv, x, m0, v0, [G.seed_of(NS_CHAIN_NOISE, s, iv.j) for s in seeds],
                              step_c=step_c, n_iterations=iters, n_grad_samples=Bg, snapshot_at=(),
                              diag_every=0, batch_size=batch_size)
        nfe += int(res["nfe"]) + int(info["nfe"])
        means, variances = res["means"], res["variances"]                      # (A, K, d), (A, K)
        ys = []
        for a, s in enumerate(seeds):                                         # uniform weights, one draw
            g = torch.Generator(device=teacher.device).manual_seed(G.seed_of(NS_CHAIN_DRAW, s, iv.j))
            y, _ = IT.sample_mixture(means[a], variances[a].float(), 1, g)
            ys.append(y[0])
        x = torch.stack(ys).reshape(len(seeds), *teacher.image_shape)
        if iv.terminal:
            x, nb = G.terminal_denoise(teacher, x, batch_size)
            nfe += int(nb)
        per_interval.append(dict(j=iv.j, variance_over_v_anc=float(variances.mean() / iv.v_anc),
                                 stopped_early=[m for m in res["stopped_early"] if m is not None],
                                 n_effective=float(np.mean([IT.component_stats(means[a], variances[a])["n_effective"]
                                                            for a in range(len(seeds))]))))
    return x, dict(variant=variant, source_variant=v.get("source_variant"), iterations=iters, step_c=step_c,
                   n_components=K, n_grad_samples=Bg, score_evals=nfe, per_interval=per_interval)


def hybrid_sample(cfg, tag_a, tag_b, mask, seeds, device="cuda", ckpt="best"):
    """Diagnostic: the standalone sampler's loop with student ``tag_a`` at the intervals where
    ``mask[j] == "1"`` and student ``tag_b`` elsewhere (e.g. trained vs untrained init), with the
    same prior and the same per-image draw generators as :class:`StudentSampler` -- so it localises
    which transitions change the end-to-end result.  Returns images (n, C, H, W) on CPU."""
    from gmmd.exp5.grid import SeededGenerators, prior_from_seeds
    from gmmd.exp5.sampler import StudentSampler
    root = C.out_dir(cfg) / "student"
    A = StudentSampler(root / tag_a / f"ckpt_{ckpt}.pt", device=device)
    B = StudentSampler(root / tag_b / f"ckpt_{ckpt}.pt", device=device)
    if len(mask) != A.n_intervals or set(mask) - {"0", "1"}:
        raise ValueError(f"mask must be {A.n_intervals} characters of 0/1, got {mask!r}")
    A._set_precision()
    seeds = [int(s) for s in seeds]
    with torch.inference_mode():
        x = prior_from_seeds(A.image_shape, A.sigma_max, seeds, A.device)
        gens = SeededGenerators(A.draw_seeds(seeds), A.device)
        n = len(seeds)
        idx = torch.arange(n, device=x.device)
        for j in range(A.n_intervals):
            net = (A if mask[j] == "1" else B).net
            logits, means, logv = net(x, j)
            p = torch.softmax(logits.float(), -1)
            k = torch.stack([torch.multinomial(p[b], 1, generator=gens.gens[b])[0] for b in range(n)])
            eps = gens.randn(A.image_shape)
            x = means[idx, k] + (0.5 * logv[idx, k]).exp().view(n, 1, 1, 1) * eps
    return x.float().cpu()
