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
    ap = argparse.ArgumentParser(prog="fitsummary", allow_abbrev=False)
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


# ======================================================================================
# cost accounting (every number read from the per-stage JSON records)
# ======================================================================================
def _sum_json(paths, keys):
    tot = {k: 0.0 for k in keys}
    n = 0
    for p in paths:
        r = json.loads(p.read_text())
        for k in keys:
            v = r
            for part in k.split("."):
                v = v.get(part, 0) if isinstance(v, dict) else 0
            tot[k] += float(v or 0)
        n += 1
    return tot, n


def stage_costs(cfg, a, rest):
    """Offline and online cost table.  GPU-seconds are wall-clock seconds of one GPU (shared with
    another user's light job); NFE = score-network examples evaluated."""
    ap = argparse.ArgumentParser(prog="costs", allow_abbrev=False)
    ap.add_argument("--student", default="main")
    o = ap.parse_args(rest)
    root = C.out_dir(cfg)
    rows = {}
    for split in ("train", "val", "test", "refine"):
        t, n = _sum_json((root / "anchors" / split).glob("traj*.json"), ["nfe", "seconds"])
        if n:
            rows[f"teacher anchor trajectories ({split}, fp32)"] = dict(n=n, nfe=t["nfe"], gpu_s=t["seconds"])
        t, n = _sum_json((root / "refs" / split).glob("traj*_j*.json"), ["nfe", "seconds", "feature_seconds"])
        if n:
            rows[f"teacher reference rollouts ({split}, fp32, evaluation only)"] = dict(
                n=n, nfe=t["nfe"], gpu_s=t["seconds"] + t["feature_seconds"])
        t, n = _sum_json((root / "endpoints" / split).glob("traj*_j*.json"), ["nfe", "seconds"])
        if n:
            rows[f"teacher endpoints ({split}, fp32, ablation data)"] = dict(n=n, nfe=t["nfe"], gpu_s=t["seconds"])
        for vdir in sorted((root / "fits" / split).glob("*")):
            t, n = _sum_json(vdir.glob("traj*_j*.json"), ["cost.fit_nfe", "cost.fit_seconds", "cost.init_nfe",
                                                          "cost.init_seconds", "bank.nfe", "bank.seconds"])
            if n:
                rows[f"IBW fits {split}/{vdir.name} (TF32; init fp32)"] = dict(
                    n=n, nfe=t["cost.fit_nfe"] + t["cost.init_nfe"] + t["bank.nfe"],
                    gpu_s=t["cost.fit_seconds"] + t["cost.init_seconds"] + t["bank.seconds"])
    for sd in sorted((root / "student").glob("*")):
        s = sd / "train_summary.json"
        if s.exists():
            r = json.loads(s.read_text())
            rows[f"student training: {sd.name} (TF32)"] = dict(n=r["steps"], nfe=0, gpu_s=r["seconds"])
    timing = {p.stem: json.loads(p.read_text()) for p in sorted((root / "timing").glob("*.json"))}
    out = dict(rows=rows, timing=timing)
    # break-even: images after which the student's offline cost is repaid by its per-image saving
    t_teacher = next((v for k, v in timing.items() if k.startswith("teacher_fp32")), None)
    t_student = next((v for k, v in timing.items() if k.startswith(f"student_{o.student}_tf32")), None)
    offline = sum(v["gpu_s"] for k, v in rows.items()
                  if "evaluation only" not in k and "ablation" not in k and "val/" not in k and "p0_" not in k
                  and (o.student in k or "training" not in k))
    if t_teacher and t_student:
        saving = t_teacher["s_per_image"] - t_student["s_per_image"]
        out["break_even"] = dict(offline_gpu_s=offline, teacher_s_per_image=t_teacher["s_per_image"],
                                 student_s_per_image=t_student["s_per_image"], saving_s_per_image=saving,
                                 images_to_amortise=offline / saving if saving > 0 else None,
                                 note="offline = train/test anchor trajectories + IBW fits used for training "
                                      "(train, refine) + student training; evaluation-only references, "
                                      "validation/pilot fits and the ablation are excluded")
    C.dump(out, root / "costs.json")
    lines = ["| item | units | NFE | GPU-hours |", "|---|---|---|---|"]
    for k, v in rows.items():
        lines.append(f"| {k} | {v['n']} | {v['nfe']:.3g} | {v['gpu_s'] / 3600:.2f} |")
    if timing:
        lines += ["", "| uncached generation | precision | s / image | images / GPU-hour | peak GB |", "|---|---|---|---|---|"]
        for k, v in timing.items():
            lines.append(f"| {v['method']} | {v['precision']} | {v['s_per_image']:.4f} | {v['images_per_gpu_hour']:.0f} | {v['peak_gb']:.1f} |")
    if "break_even" in out:
        b = out["break_even"]
        lines += ["", f"Break-even: offline {b['offline_gpu_s'] / 3600:.1f} GPU-h / saving "
                      f"{b['saving_s_per_image']:.2f} s per image = {b['images_to_amortise']:.0f} images"]
    text = "\n".join(lines)
    (root / "costs.md").write_text(text + "\n")
    print(text)


# ======================================================================================
# report tables, rendered from the result files (no hand-copied numbers)
# ======================================================================================
def table_stage_abc(cfg, tag, split="test"):
    s = json.loads((C.out_dir(cfg) / "eval" / tag / f"{split}_summary.json").read_text())
    pi = s["per_interval"]
    lines = [f"Held-out {split} anchors, {tag}: excess MMD² = MMD²(model, ref A) − MMD²(ref B, ref A) "
             "(mean ± SD over anchors; detection = permutation p < 0.05; floor detection in brackets).", "",
             "| j | A: teacher→IBW DINOv2 | B: IBW→student DINOv2 | C: teacher→student DINOv2 | A / B / C pixel | "
             "A / C detected (DINOv2) | KL(q_IBW‖q_φ)/dim | student π perplexity | IBW / student n_eff | "
             "DINOv2 dispersion ratio A / C |", "|" + "---|" * 10]
    for j in sorted(pi, key=int):
        e = pi[j]
        f = lambda st, rep: f"{e[st][rep]['excess_mean']:+.3f} ± {e[st][rep]['excess_sd']:.3f}"
        kl = e.get("kl_forward_per_dim", {})
        disp = lambda st: e[st]["dinov2"].get("dispersion_ratio_mean", float("nan"))
        lines.append(
            f"| {j} | {f('stage_A', 'dinov2')} | {f('stage_B', 'dinov2')} | {f('stage_C', 'dinov2')} | "
            f"{e['stage_A']['pixel']['excess_mean']:+.4f} / {e['stage_B']['pixel']['excess_mean']:+.4f} / "
            f"{e['stage_C']['pixel']['excess_mean']:+.4f} | "
            f"{e['stage_A']['dinov2']['detect_rate']:.2f} / {e['stage_C']['dinov2']['detect_rate']:.2f} "
            f"[{e['stage_A']['dinov2']['floor_detect_rate']:.2f}] | "
            f"{(str(round(kl['mean'], 4)) if kl else '—')} | {e['student_weight_perplexity']:.2f} | "
            f"{e['ibw_n_effective']:.1f} / {e['student_n_effective']:.1f} | {disp('stage_A'):.2f} / {disp('stage_C'):.2f} |")
    return "\n".join(lines)


def table_fid(cfg, tag):
    f = json.loads((C.out_dir(cfg) / f"fid_{tag}.json").read_text())
    lines = [f"n = {f['n_images']} images per method, same x_T seeds (1,000,000+), reference = all "
             f"{f['n_reference']} CelebA-HQ 256 images; real-vs-real FID at the same n: "
             f"{f['real_subset_fid_same_n']:.2f}. ± = bootstrap SD over images; CI = paired bootstrap 95% "
             "interval of FID(method) − FID(teacher).", "",
             "| method | FID ↓ | paired Δ vs teacher (95% CI) | precision ↑ | recall ↑ |", "|---|---|---|---|---|"]
    for m, r in f["methods"].items():
        ci = r["bootstrap"].get("diff_vs_teacher_ci95")
        lines.append(f"| {m} | {r['fid']:.2f} ± {r['bootstrap']['sd']:.2f} | "
                     f"{'—' if ci is None else f'[{ci[0]:+.1f}, {ci[1]:+.1f}]'} | {r['precision']:.3f} | {r['recall']:.3f} |")
    return "\n".join(lines)


def stage_tables(cfg, a, rest):
    ap = argparse.ArgumentParser(prog="tables", allow_abbrev=False)
    ap.add_argument("--abc", nargs="*", default=[])       # tag[:split]
    ap.add_argument("--fid", nargs="*", default=[])       # fid tags
    o = ap.parse_args(rest)
    parts = []
    for t in o.abc:
        tag, _, split = t.partition(":")
        parts += [f"### Stage A / B / C — {tag} ({split or 'test'})", "", table_stage_abc(cfg, tag, split or "test"), ""]
    for t in o.fid:
        parts += [f"### FID / precision / recall — {t}", "", table_fid(cfg, t), ""]
    text = "\n".join(parts)
    (C.out_dir(cfg) / "tables.md").write_text(text + "\n")
    print(text)
