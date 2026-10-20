"""exp5 figures, rendered from saved results only (no GPU, no teacher):

    python -m gmmd.exp5.figures --config exp5_ibw_amortized_10step_k8 [--only ibw_p0 ...]

Every figure is written as a PNG (400 dpi) + PDF pair under ``<output dir>/figures/`` by
:func:`gmmd.exp5.publication_style.save_figure` (a local copy of the academic-plotting helper).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

from gmmd.config import load_config, parse_overrides  # noqa: E402
from gmmd.exp5 import common as C  # noqa: E402
from gmmd.exp5.publication_style import (PALETTE, FigureStyle, add_panel_labels,  # noqa: E402
                                         apply_publication_style, save_figure, style_for_series)

FULL, HALF = 6.9, 3.35


def _fig_dir(cfg):
    return C.out_dir(cfg) / "figures"


def _interval_colors(n=10):
    cmap = plt.get_cmap("viridis")
    return [cmap(0.05 + 0.85 * k / max(1, n - 1)) for k in range(n)]


# ---------------------------------------------------------------------------------- IBW fits
def _load_traces(cfg, split, variant, j):
    """Per anchor: (iterations, |g|/|g_0|, diag iterations, mean drift, v/v_anc mean)."""
    out = []
    for p in sorted((C.out_dir(cfg) / "fits" / split / variant).glob(f"traj*_j{j}.npz")):
        meta = json.loads(p.with_suffix(".json").read_text())
        with np.load(p) as z:
            g = z["grad_norms"]
            out.append(dict(g=g / g[0], it_diag=z["diag_iterations"], drift=z["diag_mean_drift"],
                            v=z["diag_variances"].mean(-1) / meta["interval"]["v_anc"]))
    return out


def fig_ibw_step_size(cfg, variants=("p0_c0p2", "p0_c0p5", "p0_c1p0"), intervals=(1, 4, 7), split="val"):
    """P0: mean-gradient norm, mean drift and variance ratio vs IBW iteration for each step-size
    constant c (gamma_j = c v_anc,j), prior-free K = 8, two val anchors per interval."""
    style = apply_publication_style(FigureStyle(width_in=FULL, height_in=5.4))
    fig, axes = plt.subplots(3, len(intervals), figsize=(style.width_in, style.height_in),
                             constrained_layout=True, sharex=True)
    for col, j in enumerate(intervals):
        for i, v in enumerate(variants):
            color, ls, mk = style_for_series(i)
            c = v.split("_c")[1].replace("p", ".")
            for k, tr in enumerate(_load_traces(cfg, split, v, j)):
                lab = f"c = {c}" if k == 0 else None
                axes[0, col].semilogy(np.arange(len(tr["g"])), tr["g"], color=color, ls=ls, lw=1.0, label=lab)
                axes[1, col].semilogy(tr["it_diag"][1:], tr["drift"][1:], color=color, ls=ls, marker=mk,
                                      ms=2.5, lw=1.0, label=lab)
                axes[2, col].plot(tr["it_diag"], tr["v"], color=color, ls=ls, marker=mk, ms=2.5, lw=1.0, label=lab)
        tol = float(cfg["ibw"]["convergence"]["mean_drift_tol"])
        axes[1, col].axhline(tol, color=PALETTE["gray"], lw=0.8, ls=":")
        axes[2, col].axhline(1.0, color=PALETTE["gray"], lw=0.8, ls=":")
        axes[0, col].set_title(f"interval {j}")
        axes[2, col].set_xlabel("IBW iteration")
    axes[0, 0].set_ylabel(r"$\|\nabla_m F\| / \|\nabla_m F\|_0$")
    axes[1, 0].set_ylabel("mean drift / 25 it")
    axes[2, 0].set_ylabel(r"mean $v_k / v_{\rm anc}$")
    for ax in axes[2]:
        ax.set_ylim(0.5, 2.5)
    axes[0, 0].legend(loc="upper right")
    add_panel_labels(axes[:, 0], ["(a)", "(b)", "(c)"], x=-0.32)
    return save_figure(fig, _fig_dir(cfg) / "fig_p0_ibw_step_size")


def fig_stage_a_vs_iterations(cfg, variants, split="val", basename="fig_p1_stage_a_vs_iterations"):
    """Stage A (teacher -> IBW) DINOv2 excess MMD^2 vs IBW iteration: mean over all evaluated
    (anchor, interval) pairs (left) and per interval (small multiples, one per interval)."""
    from gmmd.exp5.report import fit_rows
    data = {v: fit_rows(cfg, split, v) for v in variants}
    js = sorted({r["j"] for rows in data.values() for r in rows})
    ncol = (len(js) + 1) // 2
    style = apply_publication_style(FigureStyle(width_in=FULL, height_in=4.2))
    fig = plt.figure(figsize=(style.width_in, style.height_in), constrained_layout=True)
    gs = fig.add_gridspec(2, 2 + ncol, width_ratios=[1.15, 1.15] + [1] * ncol)
    ax0 = fig.add_subplot(gs[:, :2])
    small = [fig.add_subplot(gs[r, 2 + c]) for r in range(2) for c in range(ncol)][:len(js)]
    all_its = set()
    n_pairs = 0
    for i, v in enumerate(variants):
        color, ls, mk = style_for_series(i)
        rows = [r for r in data[v] if "snapshots" in r]
        if not rows:
            continue
        n_pairs = max(n_pairs, len(rows))
        its = sorted({int(t) for r in rows for t in r["snapshots"]})
        all_its |= set(its)
        mean = [np.nanmean([r["snapshots"][str(t)]["dino"] for r in rows if str(t) in r["snapshots"]]) for t in its]
        ax0.plot(its, mean, color=color, ls=ls, marker=mk, label=v)
        for ax, j in zip(small, js):
            rj = [r for r in rows if r["j"] == j]
            if rj:
                m = [np.nanmean([r["snapshots"][str(t)]["dino"] for r in rj if str(t) in r["snapshots"]]) for t in its]
                ax.plot(its, m, color=color, ls=ls, marker=mk, ms=2.5, lw=1.0)
    ticks = sorted(all_its)
    labelled = [t for t in ticks if t != 200] if {100, 200, 300} <= set(ticks) else ticks   # 200|300 collide on log axes
    for ax in [ax0] + small:
        ax.set_xscale("log")
        ax.set_xticks(labelled)
        ax.set_xticks([t for t in ticks if t not in labelled], minor=True)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax0.set_xlabel("IBW iteration")
    ax0.set_ylabel("DINOv2 excess MMD$^2$ (IBW vs teacher)")
    ax0.text(0.06, 0.50, f"mean over {n_pairs} {split}\n(anchor, interval) pairs", transform=ax0.transAxes,
             va="top", fontsize=7.5)
    ax0.axhline(0, color=PALETTE["gray"], lw=0.8, ls=":")
    ax0.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06))
    for k, (ax, j) in enumerate(zip(small, js)):
        ax.set_title(f"j = {j}", fontsize=8)
        ax.axhline(0, color=PALETTE["gray"], lw=0.6, ls=":")
        ax.tick_params(labelsize=6)
        ax.set_ylim(bottom=0)
        if k // ncol == 1:
            ax.set_xlabel("iteration", fontsize=7)
    return save_figure(fig, _fig_dir(cfg) / basename)


def fig_posthoc_fine_grid(cfg, split="val", basename="fig_posthoc_ibw_iterations"):
    """Post-hoc sensitivity of the frozen iteration count: stage-A excess (DINOv2, pixels) of the
    prior-free initialisation against IBW iteration, fresh-seed fits on a 5-50 grid (lines; mean
    and paired-bootstrap 95 % CI over the val pairs) with the pre-registered P1 fits (markers)."""
    from gmmd.exp5.report import fit_rows
    sel = json.loads((C.out_dir(cfg) / "selection.json").read_text())
    fine = [r for r in fit_rows(cfg, split, "prior_free_fine") if "snapshots" in r]
    p1 = [r for r in fit_rows(cfg, split, sel["variant"]) if "snapshots" in r]
    style = apply_publication_style(FigureStyle(width_in=FULL, height_in=2.5))
    fig, axes = plt.subplots(1, 2, figsize=(style.width_in, style.height_in), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, key, lab in zip(axes, ("dino", "pixel"), ("DINOv2 excess MMD$^2$", "pixel excess MMD$^2$")):
        its = sorted({int(t) for r in fine for t in r["snapshots"]})
        vals = np.array([[r["snapshots"][str(t)][key] for t in its] for r in fine])      # (pairs, its)
        boot = vals[rng.integers(0, len(vals), (4000, len(vals)))].mean(1)
        color, ls, mk = style_for_series(0)
        ax.fill_between(its, np.quantile(boot, 0.025, 0), np.quantile(boot, 0.975, 0), color=color, alpha=0.2, lw=0)
        ax.plot(its, vals.mean(0), color=color, ls=ls, marker=mk, label="post-hoc fine grid (fresh seeds)")
        its1 = sorted({int(t) for r in p1 for t in r["snapshots"]})
        m1 = [np.mean([r["snapshots"][str(t)][key] for r in p1]) for t in its1]
        c2, _, mk2 = style_for_series(1)
        ax.plot(its1, m1, color=c2, ls="none", marker=mk2, label="pre-registered P1 snapshots")
        ax.axvline(sel["iterations"], color=PALETTE["gray"], lw=0.8, ls="--")
        ax.set_xscale("log")
        ticks = [5, 10, 20, 50, 100, 300]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel("IBW iteration")
        ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
    for ax, y in zip(axes, (0.03, 0.90)):                 # where each panel has no data near the line
        ax.text(sel["iterations"] * 1.07, ax.get_ylim()[1] * y, f"frozen: {sel['iterations']} it", fontsize=7,
                va="bottom", color=PALETTE["gray"])
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].set_title(f"mean over {len(fine)} {split} pairs (band: 95 % bootstrap CI)", fontsize=8)
    return save_figure(fig, _fig_dir(cfg) / basename)


# ---------------------------------------------------------------------------------- stage A/B/C
STAGE_STYLE = {"stage_A": ("A: teacher → IBW", 0), "stage_B": ("B: IBW → student", 1),
               "stage_C": ("C: teacher → student", 2)}


def fig_stage_abc(cfg, tag, split="test"):
    """Per-interval excess MMD^2 (model vs reference minus reference vs reference, mean ± SD over
    held-out anchors) for the three approximation stages, DINOv2 (a) and pixels (b), with the
    IBW-vs-IBW floor of stage B (an independent replicate fit scored like the student), plus the
    stage-B forward KL per dimension and the same KL between the two IBW fits (c)."""
    summ = json.loads((C.out_dir(cfg) / "eval" / tag / f"{split}_summary.json").read_text())
    pi = summ["per_interval"]
    js = sorted(int(j) for j in pi)
    style = apply_publication_style(FigureStyle(width_in=FULL, height_in=2.8))
    fig, axes = plt.subplots(1, 3, figsize=(style.width_in, style.height_in), layout="constrained")
    for st, (lab, i) in STAGE_STYLE.items():
        color, ls, mk = style_for_series(i)
        for ax, rep in zip(axes[:2], ("dinov2", "pixel")):
            m = np.array([pi[str(j)][st][rep]["excess_mean"] for j in js])
            sd = np.array([pi[str(j)][st][rep]["excess_sd"] for j in js])
            off = (i - 1) * 0.12
            ax.errorbar(np.array(js) + off, m, yerr=sd, color=color, ls=ls, marker=mk, ms=3.5, lw=1.0,
                        capsize=1.5, label=lab)
    have_rep = all(pi[str(j)].get("stage_B_rep") for j in js)
    if have_rep:
        for ax, rep in zip(axes[:2], ("dinov2", "pixel")):
            ax.plot(js, [pi[str(j)]["stage_B_rep"][rep]["excess_mean"] for j in js], color=PALETTE["gray"], lw=1.0,
                    ls="--", marker="x", ms=3, label="B floor: IBW replicate → IBW")
    kl = [pi[str(j)].get("kl_forward_per_dim", {}).get("mean", np.nan) for j in js]
    color, ls, mk = style_for_series(1)
    axes[2].plot(js, kl, color=color, ls=ls, marker=mk, label=r"KL$(q_{\rm IBW}\,\|\,q_\phi)$ / dim")
    klr = [pi[str(j)].get("rep_kl_forward_per_dim", {}).get("mean", np.nan) for j in js]
    if np.isfinite(klr).any():
        axes[2].plot(js, klr, color=PALETTE["gray"], ls="--", marker="x", ms=3,
                     label=r"KL$(q_{\rm IBW}\,\|\,q_{\rm IBW\,replicate})$ / dim")
    for ax in axes:
        ax.axhline(0, color=PALETTE["gray"], lw=0.7, ls=":")
        ax.set_xlabel("interval j")
        ax.set_xticks(js)
    axes[0].set_ylabel("excess MMD$^2$ (DINOv2)")
    axes[1].set_ylabel("excess MMD$^2$ (pixels)")
    axes[2].set_ylabel("nats per dimension")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(l), fontsize=6.5, frameon=False)
    axes[2].legend(loc="upper left", fontsize=6)
    add_panel_labels(axes, x=-0.28)
    return save_figure(fig, _fig_dir(cfg) / f"fig_stage_abc_{tag}_{split}")


def fig_kernels_vs_teacher(cfg, tags, labels, split="test", basename=None):
    """Per interval, how far each one-step kernel is from the teacher's 200-step kernel at the same
    held-out anchors: stage A of the IBW targets and stage C of every student tag (excess MMD^2,
    mean ± SD over anchors), DINOv2 (a) and pixels (b)."""
    S = {t: json.loads((C.out_dir(cfg) / "eval" / t / f"{split}_summary.json").read_text())["per_interval"] for t in tags}
    js = sorted(int(j) for j in S[tags[0]])
    series = [("IBW targets (stage A)", S[tags[0]], "stage_A")] + [(lab, S[t], "stage_C") for t, lab in zip(tags, labels)]
    style = apply_publication_style(FigureStyle(width_in=FULL, height_in=2.6))
    fig, axes = plt.subplots(1, 2, figsize=(style.width_in, style.height_in), constrained_layout=True)
    k = len(series)
    for i, (lab, pi, st) in enumerate(series):
        color, ls, mk = style_for_series(i)
        off = (i - (k - 1) / 2) * 0.1
        for ax, rep in zip(axes, ("dinov2", "pixel")):
            m = np.array([pi[str(j)][st][rep]["excess_mean"] for j in js])
            sd = np.array([pi[str(j)][st][rep]["excess_sd"] for j in js])
            ax.errorbar(np.array(js) + off, m, yerr=sd, color=color, ls=ls, marker=mk, ms=3.2, lw=1.0, capsize=1.2,
                        label=lab)
    for ax, rep in zip(axes, ("DINOv2", "pixels")):
        ax.axhline(0, color=PALETTE["gray"], lw=0.7, ls=":")
        ax.set_xlabel("interval j")
        ax.set_xticks(js)
        ax.set_ylabel(f"excess MMD$^2$ ({rep})")
        ax.set_title("one-step kernel vs the teacher's 200 steps", fontsize=7.5)
    axes[1].legend(loc="upper left", fontsize=6)
    add_panel_labels(axes, x=-0.16, y=1.06)
    return save_figure(fig, _fig_dir(cfg) / (basename or f"fig_kernels_vs_teacher_{split}"))


# ---------------------------------------------------------------------------------- training
def fig_detail_per_interval(cfg, split="val", kinds=("ibw", "init_main", "main"), labels=None, basename=None):
    """Contrast and fine-detail (3x3 Laplacian RMS) of each kernel's Tweedie-denoised end states,
    as a ratio to the teacher's 200-step kernel from the same anchor (mean ± SD over anchors), from
    ``figures/diag/detail_per_interval_<split>.json`` (:mod:`gmmd.exp5.diag_detail`)."""
    d = json.loads((_fig_dir(cfg) / "diag" / f"detail_per_interval_{split}.json").read_text())
    rows = d["rows"]
    kinds = [k for k in kinds if k in rows[0]]
    labels = labels or {"ibw": "IBW targets (cached fits)", "init_main": "untrained init (Tweedie Gaussian)",
                        "main": "IBW-GMMD student", "refined": "IBW-GMMD student, refined",
                        "ablation_endpoints": "ablation: endpoint NLL"}
    js = sorted({r["j"] for r in rows})
    style = apply_publication_style(FigureStyle(width_in=FULL, height_in=2.5))
    fig, axes = plt.subplots(1, 2, figsize=(style.width_in, style.height_in), layout="constrained", sharex=True)
    for ax, stat, name in zip(axes, ("contrast", "detail"), ("contrast (pixel SD)", "detail (Laplacian RMS)")):
        ax.axhline(1.0, color=PALETTE["black"], lw=0.8, ls=":")
        ax.axvspan(-0.5, 0.5, color="0.9", lw=0)
        for i, k in enumerate(kinds):
            color, ls, mk = style_for_series(i)
            r = np.array([[x[k][stat] / x["teacher"][stat] for x in rows if x["j"] == j] for j in js])
            ax.errorbar(np.array(js) + 0.08 * (i - (len(kinds) - 1) / 2), r.mean(1), yerr=r.std(1, ddof=1),
                        color=color, ls=ls, marker=mk, ms=3.5, lw=1.0, capsize=2, label=labels.get(k, k))
        ax.set_xticks(js)
        ax.set_xlabel("interval j (0 = highest noise)")
        ax.set_ylabel(f"{name}\nkernel / teacher")
        ax.set_ylim(0, 1.45)
    n_anc = len({r["n"] for r in rows})
    axes[0].text(0, 0.04, "denoising\nartifact", ha="center", va="bottom", fontsize=5.5, color="0.35")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(kinds), fontsize=6.5,
               frameon=False)
    add_panel_labels(axes, x=-0.2, y=1.04)
    print(f"detail figure: {split}, {n_anc} anchors per interval, {d['n_per_set']} samples per kernel")
    return save_figure(fig, _fig_dir(cfg) / (basename or f"fig_detail_per_interval_{split}"))


def fig_training_curves(cfg, tag):
    rows = [json.loads(l) for l in (C.out_dir(cfg) / "student" / tag / "curves.jsonl").read_text().splitlines()]
    vals = [r for r in rows if "val" in r]
    trains = [r for r in rows if "loss" in r]
    js = sorted(int(k) for k in vals[0]["val"] if k != "mean_nll")
    cols = _interval_colors(len(js))
    style = apply_publication_style(FigureStyle(width_in=FULL, height_in=2.5))
    fig, axes = plt.subplots(1, 3, figsize=(style.width_in, style.height_in), constrained_layout=True)
    st = [r["step"] for r in vals]
    for j, c in zip(js, cols):
        axes[0].plot(st, [r["val"][str(j)]["nll"] - vals[0]["val"][str(j)]["nll"] for r in vals], color=c, lw=1.0,
                     label=f"j={j}")
        k = [r["val"][str(j)].get("kl", np.nan) for r in vals]
        if np.isfinite(k).any():
            axes[1].plot(st, k, color=c, lw=1.0)
    endpoints = not any("kl" in r["val"][str(j)] for r in vals for j in js)
    if endpoints:                                   # the ablation: no IBW density, no KL to report
        axes[1].text(0.5, 0.5, "not applicable:\ntrained and validated on\nteacher endpoints (no IBW)",
                     transform=axes[1].transAxes, ha="center", va="center", fontsize=7, color=PALETTE["black"])
        axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[2].plot([r["step"] for r in trains], [r["loss"] for r in trains], color=PALETTE["black"], lw=1.0,
                 label="train")
    axes[2].plot(st, [r["val"]["mean_nll"] for r in vals], color=PALETTE["vermillion"], ls="--", lw=1.0,
                 label="val, EMA (endpoints)" if endpoints else "val, EMA (IBW draws)")
    axes[0].set_ylabel("val NLL change (nats/dim)")
    axes[1].set_ylabel(r"val KL$(q_{\rm IBW}\|q_\phi)$ (nats/dim)")
    axes[2].set_ylabel("NLL (standardised, nats/dim)")
    for ax in axes:
        ax.set_xlabel("training step")
    axes[0].legend(ncol=2, fontsize=5.5, loc="lower left")
    lo, hi = axes[2].get_ylim()
    axes[2].set_ylim(lo, hi + 0.35 * (hi - lo))           # headroom: the legend sits above the curves
    axes[2].legend(fontsize=6.5, loc="upper right")
    add_panel_labels(axes, x=-0.3)
    return save_figure(fig, _fig_dir(cfg) / f"fig_training_{tag}")


# ---------------------------------------------------------------------------------- samples
def _first_images(cfg, method, n):
    from gmmd.exp5.evaluation import _images
    return _images(cfg, method, n)[0]


def fig_samples(cfg, methods, labels, basename, n_grid=16, n_pair=8):
    """(top) the first ``n_grid`` images of each method (no selection), (bottom) the same x_T
    across methods for the first ``n_pair`` seeds."""
    imgs = {m: _first_images(cfg, m, max(n_grid, n_pair)) for m in methods}
    label_in = 0.95                                       # row labels; rows sized from the image aspect
    style = apply_publication_style(FigureStyle(width_in=FULL, height_in=len(methods) * (FULL - label_in) / n_pair + 0.1))
    fig, axes = plt.subplots(len(methods), 1, figsize=(style.width_in, style.height_in),
                             gridspec_kw=dict(left=label_in / FULL, right=0.995, top=0.99, bottom=0.01, hspace=0.04))
    for ax, m, lab in zip(np.atleast_1d(axes), methods, labels):
        row = np.concatenate([np.pad(im, ((2, 2), (2, 2), (0, 0)), constant_values=255)
                              for im in imgs[m][:n_pair]], axis=1)
        ax.imshow(row, interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=7)
    return save_figure(fig, _fig_dir(cfg) / basename, dpi=600)


def save_grid(cfg, method, n=64, basename=None):
    """Plain 8x8 mosaic PNG of the first n images (no selection), for the report."""
    from PIL import Image
    im = _first_images(cfg, method, n)
    k = int(np.sqrt(n))
    g = im[:k * k].reshape(k, k, *im.shape[1:]).transpose(0, 2, 1, 3, 4).reshape(k * im.shape[1], k * im.shape[2], 3)
    path = _fig_dir(cfg) / (basename or f"grid_{method.replace(':', '_')}_{n}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(g).save(path)
    return path, path


# ---------------------------------------------------------------------------------- quality vs cost
METHOD_LABELS = {"teacher": "teacher (2,000-step SMLD)", "student:main": "IBW-GMMD student",
                 "student:refined": "IBW-GMMD student, refined", "student:ablation_endpoints": "ablation: endpoint NLL",
                 "student:init_main": "untrained init (Tweedie + noise)", "student:pilot": "pilot student",
                 "student:init_pilot": "untrained init (pilot)",
                 "ibwchain": "IBW chain (online IBW, no network)",
                 "student:main:best:meanlast": "IBW-GMMD student, noise-free last step (diagnostic)",
                 "student:refined:best:meanlast": "refined, noise-free last step (diagnostic)"}


def method_label(m):
    if m in METHOD_LABELS:
        return METHOD_LABELS[m]
    if m.startswith("coarse:"):
        _, kind, k = m.split(":")
        return f"{dict(ancestral='ancestral', reverse_diffusion='SMLD').get(kind, kind)} {k} steps"
    return m


def nfe_per_image(cfg, method):
    """Network evaluations per image, from the generation records (teacher: exp4's 2,000 steps)."""
    if method == "teacher":
        return int(json.loads((C.out_dir(cfg) / "partition.json").read_text())["n_grid"])
    from gmmd.exp5.evaluation import _method_dir
    recs = [json.loads(p.read_text()) for p in sorted(_method_dir(cfg, method).glob("chunk*.json"))]
    return sum(r["network_evals"] for r in recs) / sum(r["n"] for r in recs)


def _load_fids(cfg, fid_tags):
    """Methods of one or more FID files; they must share n, reference and preprocessing."""
    files = [json.loads((C.out_dir(cfg) / f"fid_{t}.json").read_text()) for t in fid_tags]
    keys = ("n_images", "n_reference", "fid_extractor", "preprocessing")
    for f in files[1:]:
        bad = [k for k in keys if f[k] != files[0][k]]
        if bad:
            raise ValueError(f"FID files {fid_tags} differ in {bad}; refusing to plot them together")
    methods = {}
    for f in files:
        methods.update(f["methods"])
    return files[0], methods


def fig_quality_vs_nfe(cfg, fid_tags, basename=None, order=None):
    """FID (n images, ± paired-bootstrap SD) against network evaluations per image (log x).
    One series per method; the coarse ancestral sampler at several step counts is one line."""
    head, methods = _load_fids(cfg, fid_tags)
    order = [m for m in (order or methods) if m in methods]
    style = apply_publication_style(FigureStyle(width_in=HALF, height_in=2.8))
    fig, ax = plt.subplots(figsize=(style.width_in, style.height_in), layout="constrained")
    anc = sorted((m for m in order if m.startswith("coarse:ancestral")), key=lambda m: int(m.split(":")[2]))
    singles = [m for m in order if m not in anc]
    nfe = {m: nfe_per_image(cfg, m) for m in anc + singles}
    x = {}
    for v in sorted(set(nfe.values())):                  # dodge methods sharing an evaluation count
        grp = [m for m in anc + singles if nfe[m] == v]
        for k, m in enumerate(grp):
            x[m] = v * 10 ** (0.035 * (k - (len(grp) - 1) / 2))
    i = 0
    if anc:
        color, ls, mk = style_for_series(i)
        i += 1
        ax.errorbar([x[m] for m in anc], [methods[m]["fid"] for m in anc],
                    yerr=[methods[m]["bootstrap"]["sd"] for m in anc], color=color, ls=ls, marker=mk, ms=4, lw=1.0,
                    capsize=2, label=f"ancestral ({', '.join(m.split(':')[2] for m in anc)} steps)")
    for m in singles:
        color, _, mk = style_for_series(i)
        i += 1
        ax.errorbar([x[m]], [methods[m]["fid"]], yerr=[methods[m]["bootstrap"]["sd"]],
                    color=color, ls="none", marker=mk, ms=4, capsize=2, label=method_label(m))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([t for t in (20, 30, 50, 100, 200, 400) if t < 1.3 * max(r["fid"] for r in methods.values())])
    ax.set_xlabel("network evaluations per image")
    ax.set_ylabel(f"FID (n = {head['n_images']:,}; log scale)")
    fig.legend(fontsize=6, loc="outside lower center", ncol=2, frameon=False)
    return save_figure(fig, _fig_dir(cfg) / (basename or f"fig_quality_vs_nfe_{'_'.join(fid_tags)}"))


FIGURES = {"ibw_p0": lambda cfg: fig_ibw_step_size(cfg),
           "stage_a_p0": lambda cfg: fig_stage_a_vs_iterations(cfg, ["p0_c0p2", "p0_c0p5", "p0_c1p0"],
                                                                basename="fig_p0_stage_a_vs_iterations"),
           "stage_a_p1": lambda cfg: fig_stage_a_vs_iterations(cfg, [v["name"] for v in cfg["ibw"]["variants"]]),
           "posthoc_fine": lambda cfg: fig_posthoc_fine_grid(cfg),
           "detail_val": lambda cfg: fig_detail_per_interval(cfg, "val"),
           "detail_test": lambda cfg: fig_detail_per_interval(
               cfg, "test", kinds=("ibw", "init_main", "main", "refined", "ablation_endpoints"))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="exp5_ibw_amortized_10step_k8")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    for name, f in FIGURES.items():
        if a.only and name not in a.only:
            continue
        try:
            png, pdf = f(cfg)
            print(f"{name}: {png}")
        except Exception as e:                       # one missing input must not stop the rest
            print(f"{name}: skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
