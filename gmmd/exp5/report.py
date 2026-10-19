"""exp5 summaries: fit diagnostics and stage-A tables (P0/P1), and report figures.

    python -m gmmd.experiments.ibw_amortized fitsummary --split val --set ... -- --variants p0_c0p2 p0_c0p5
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from gmmd.exp5 import common as C

__all__ = ["fit_rows", "stage_fitsummary"]


def stationarity(npz_path):
    """Secondary (post-hoc, clearly labelled) convergence diagnostic from the gradient-norm trace:
    the mean |grad_m F| over the last quarter of the run divided by its mean over the preceding
    quarter (~1 once the iterates fluctuate around a fixed point at the Monte-Carlo noise floor),
    and the overall reduction |g|_end / |g|_0.  ``stationary`` iff the ratio is in [0.8, 1.25]
    and the gradient fell at least 10x.  Unlike the pre-registered drift test it does not depend
    on the noise floor's absolute size, which grows at low-noise intervals."""
    with np.load(npz_path) as z:
        g = np.asarray(z["grad_norms"], float)
    n = len(g)
    late, prev = g[3 * n // 4:].mean(), g[n // 2:3 * n // 4].mean()
    ratio = float(late / max(prev, 1e-300))
    red = float(g[-25:].mean() / max(g[0], 1e-300))
    return dict(grad_plateau_ratio=ratio, grad_reduction=red,
                stationary=bool(0.8 <= ratio <= 1.25 and red < 0.1))


def fit_rows(cfg, split, variant):
    """One row per fit: convergence, failures, scale and component statistics, and (if evaluated)
    the stage-A DINOv2 / pixel excess at every snapshot."""
    root = C.out_dir(cfg)
    rows = []
    for p in sorted((root / "fits" / split / variant).glob("traj*_j*.json")):
        m = json.loads(p.read_text())
        ev = root / "fits_eval" / split / variant / p.name
        e = json.loads(ev.read_text()) if ev.exists() else None
        st = stationarity(p.with_suffix(".npz"))
        r = dict(variant=variant, n=m["n"], j=m["j"], stopped=m["stopped_early"],
                 converged=m["convergence"]["converged"], converged_from=m["convergence"]["converged_from"],
                 final_mean_drift=m["convergence"]["final_mean_drift"],
                 final_logvar_drift=m["convergence"]["final_logvar_drift"],
                 grad_ratio=m["grad_norm_last"] / max(m["grad_norm_first"], 1e-30),
                 v_over_vanc=float(np.mean(m["variances_over_v_anc"])),
                 n_eff=m["component_stats"]["n_effective"], sep_min=m["component_stats"]["sep_min"],
                 sep_median=m["component_stats"]["sep_median"], init_sep_median=m["init"]["stats"]["sep_median"],
                 fit_seconds=m["cost"]["fit_seconds"], fit_nfe=m["cost"]["fit_nfe"],
                 init_nfe=m["cost"]["init_nfe"], iterations=m["n_iterations"], **st)
        if e is not None:
            r["snapshots"] = {it: dict(dino=s["dinov2"]["excess"], pixel=s["pixel"]["excess"],
                                       p_dino=s["dinov2"]["p_model"], p_floor=s["dinov2"]["p_floor"],
                                       n_eff=s["stats"]["n_effective"], v_over_vanc=s["variance_over_v_anc"])
                              if s.get("finite") else dict(dino=float("nan"), pixel=float("nan"))
                              for it, s in e["snapshots"].items()}
        rows.append(r)
    return rows


def _fmt(x, f="{:+.4f}"):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def stage_fitsummary(cfg, a, rest):
    ap = argparse.ArgumentParser(prog="fitsummary")
    ap.add_argument("--variants", nargs="+", required=True)
    o = ap.parse_args(rest)
    lines = []
    for v in o.variants:
        rows = fit_rows(cfg, a.split, v)
        if not rows:
            continue
        its = sorted({int(i) for r in rows for i in r.get("snapshots", {})})
        lines.append(f"\n### {a.split} / {v}: {len(rows)} fits, {sum(r['stopped'] is not None for r in rows)} stopped early, "
                     f"{sum(r['converged'] for r in rows)} converged by the pre-registered drift test, "
                     f"{sum(r['stationary'] for r in rows)} stationary by the gradient-plateau test")
        hdr = "| j | n | conv. from | drift(m) | drift(logv) | |g|_T/|g|_0 | v/v_anc | n_eff | sep med (init→final) | " + \
              " | ".join(f"X@{it}" for it in its) + " | pix X@last | s/fit |"
        lines += [hdr, "|" + "---|" * (10 + len(its) + 1)]
        for r in sorted(rows, key=lambda r: (r["j"], r["n"])):
            sn = r.get("snapshots", {})
            xs = " | ".join(_fmt(sn.get(str(it), {}).get("dino")) for it in its)
            lastpix = _fmt(sn.get(str(its[-1]), {}).get("pixel"), "{:+.5f}") if its else "—"
            lines.append(f"| {r['j']} | {r['n']} | {r['converged_from'] or '—'} | {r['final_mean_drift']:.3f} | "
                         f"{_fmt(r['final_logvar_drift'], '{:.4f}')} | {r['grad_ratio']:.3f} | {r['v_over_vanc']:.3f} | "
                         f"{r['n_eff']} | {r['init_sep_median']:.1f}→{r['sep_median']:.1f} | {xs} | {lastpix} | {r['fit_seconds']:.0f} |")
        if its:
            means = []
            for it in its:
                vals = [r["snapshots"][str(it)]["dino"] for r in rows if str(it) in r.get("snapshots", {})]
                means.append(f"{np.nanmean(vals):+.4f}")
            lines.append(f"| mean | | | | | | | | | {' | '.join(means)} | | |")
    text = "\n".join(lines)
    print(text)
    (C.out_dir(cfg) / f"fitsummary_{a.split}_{'_'.join(o.variants)}.md").write_text(text + "\n")
