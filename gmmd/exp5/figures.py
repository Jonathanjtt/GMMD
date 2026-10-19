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
    (anchor, interval) pairs (left) and per interval (small multiples)."""
    from gmmd.exp5.report import fit_rows
    data = {v: fit_rows(cfg, split, v) for v in variants}
    js = sorted({r["j"] for rows in data.values() for r in rows})
    style = apply_publication_style(FigureStyle(width_in=FULL, height_in=4.6))
    fig = plt.figure(figsize=(style.width_in, style.height_in), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)
    ax0 = fig.add_subplot(gs[:, :2])
    small = [fig.add_subplot(gs[r, 2 + c]) for r in range(2) for c in range(4)][:len(js)]
    for i, v in enumerate(variants):
        color, ls, mk = style_for_series(i)
        rows = [r for r in data[v] if "snapshots" in r]
        if not rows:
            continue
        its = sorted({int(t) for r in rows for t in r["snapshots"]})
        mean = [np.nanmean([r["snapshots"][str(t)]["dino"] for r in rows if str(t) in r["snapshots"]]) for t in its]
        ax0.plot(its, mean, color=color, ls=ls, marker=mk, label=v)
        for ax, j in zip(small, js):
            rj = [r for r in rows if r["j"] == j]
            if rj:
                m = [np.nanmean([r["snapshots"][str(t)]["dino"] for r in rj if str(t) in r["snapshots"]]) for t in its]
                ax.plot(its, m, color=color, ls=ls, marker=mk, ms=2.5, lw=1.0)
    ax0.set_xscale("log")
    ax0.set_xlabel("IBW iteration")
    ax0.set_ylabel("DINOv2 excess MMD$^2$ (IBW vs teacher)")
    ax0.axhline(0, color=PALETTE["gray"], lw=0.8, ls=":")
    ax0.legend(loc="upper right")
    for ax, j in zip(small, js):
        ax.set_xscale("log")
        ax.set_title(f"j = {j}", fontsize=8)
        ax.axhline(0, color=PALETTE["gray"], lw=0.6, ls=":")
        ax.tick_params(labelsize=6.5)
    return save_figure(fig, _fig_dir(cfg) / basename)


# ---------------------------------------------------------------------------------- stage A/B/C
STAGE_STYLE = {"stage_A": ("A: teacher → IBW", 0), "stage_B": ("B: IBW → student", 1),
               "stage_C": ("C: teacher → student", 2)}


def fig_stage_abc(cfg, tag, split="test"):
    """Per-interval excess MMD^2 (model vs reference minus reference vs reference, mean ± SD over
    held-out anchors) for the three approximation stages, DINOv2 (a) and pixels (b), plus the
    stage-B forward KL per dimension (c)."""
    summ = json.loads((C.out_dir(cfg) / "eval" / tag / f"{split}_summary.json").read_text())
    pi = summ["per_interval"]
    js = sorted(int(j) for j in pi)
    style = apply_publication_style(FigureStyle(width_in=FULL, height_in=2.5))
    fig, axes = plt.subplots(1, 3, figsize=(style.width_in, style.height_in), constrained_layout=True)
    for st, (lab, i) in STAGE_STYLE.items():
        color, ls, mk = style_for_series(i)
        for ax, rep in zip(axes[:2], ("dinov2", "pixel")):
            m = np.array([pi[str(j)][st][rep]["excess_mean"] for j in js])
            sd = np.array([pi[str(j)][st][rep]["excess_sd"] for j in js])
            off = (i - 1) * 0.12
            ax.errorbar(np.array(js) + off, m, yerr=sd, color=color, ls=ls, marker=mk, ms=3.5, lw=1.0,
                        capsize=1.5, label=lab)
    kl = [pi[str(j)].get("kl_forward_per_dim", {}).get("mean", np.nan) for j in js]
    color, ls, mk = style_for_series(1)
    axes[2].plot(js, kl, color=color, ls=ls, marker=mk, label=r"KL$(q_{\rm IBW}\,\|\,q_\phi)$ / dim")
    for ax in axes:
        ax.axhline(0, color=PALETTE["gray"], lw=0.7, ls=":")
        ax.set_xlabel("interval j")
        ax.set_xticks(js)
    axes[0].set_ylabel("excess MMD$^2$ (DINOv2)")
    axes[1].set_ylabel("excess MMD$^2$ (pixels)")
    axes[2].set_ylabel("nats per dimension")
    axes[0].legend(loc="upper left", fontsize=6.5)
    axes[2].legend(loc="upper left", fontsize=6.5)
    add_panel_labels(axes, x=-0.28)
    return save_figure(fig, _fig_dir(cfg) / f"fig_stage_abc_{tag}_{split}")


# ---------------------------------------------------------------------------------- training
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
    axes[2].plot([r["step"] for r in trains], [r["loss"] for r in trains], color=PALETTE["black"], lw=1.0,
                 label="train (running mean)")
    axes[2].plot(st, [r["val"]["mean_nll"] for r in vals], color=PALETTE["vermillion"], ls="--", lw=1.0,
                 label="val (EMA)")
    axes[0].set_ylabel("val NLL change (nats/dim)")
    axes[1].set_ylabel(r"val KL$(q_{\rm IBW}\|q_\phi)$ (nats/dim)")
    axes[2].set_ylabel("NLL (standardised, nats/dim)")
    for ax in axes:
        ax.set_xlabel("training step")
    axes[0].legend(ncol=2, fontsize=5.5, loc="lower left")
    axes[2].legend(fontsize=6.5)
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
    style = apply_publication_style(FigureStyle(width_in=FULL, height_in=1.25 * len(methods) + 0.4))
    fig, axes = plt.subplots(len(methods), 1, figsize=(style.width_in, style.height_in), constrained_layout=True)
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
def fig_quality_vs_nfe(cfg, fid_tag, timing=None):
    """FID (n images, ± bootstrap SD) against network evaluations per image (log scale)."""
    f = json.loads((C.out_dir(cfg) / f"fid_{fid_tag}.json").read_text())
    style = apply_publication_style(FigureStyle(width_in=HALF, height_in=2.6))
    fig, ax = plt.subplots(figsize=(style.width_in, style.height_in), constrained_layout=True)
    groups = {"teacher (2,000-step SMLD)": [], "coarse ancestral": [], "coarse SMLD": [], "IBW-GMMD student": [],
              "ablation (endpoint NLL)": []}
    for m, r in f["methods"].items():
        if m == "teacher":
            groups["teacher (2,000-step SMLD)"].append((2000, r))
        elif m.startswith("coarse:ancestral"):
            groups["coarse ancestral"].append((int(m.split(":")[2]), r))
        elif m.startswith("coarse:reverse_diffusion"):
            groups["coarse SMLD"].append((int(m.split(":")[2]), r))
        elif "ablation" in m:
            groups["ablation (endpoint NLL)"].append((10, r))
        elif m.startswith("student"):
            groups["IBW-GMMD student"].append((10, r))
    for i, (g, pts) in enumerate(groups.items()):
        if not pts:
            continue
        pts.sort(key=lambda t: t[0])
        color, ls, mk = style_for_series(i)
        x = [p[0] for p in pts]
        y = [p[1]["fid"] for p in pts]
        e = [p[1]["bootstrap"]["sd"] for p in pts]
        ax.errorbar(x, y, yerr=e, color=color, ls=ls if len(pts) > 1 else "none", marker=mk, ms=4, lw=1.0,
                    capsize=2, label=g)
    ax.set_xscale("log")
    ax.set_xlabel("network evaluations per image")
    ax.set_ylabel(f"FID (n = {f['n_images']})")
    ax.legend(fontsize=6, loc="upper right")
    return save_figure(fig, _fig_dir(cfg) / f"fig_quality_vs_nfe_{fid_tag}")


FIGURES = {"ibw_p0": lambda cfg: fig_ibw_step_size(cfg),
           "stage_a_p0": lambda cfg: fig_stage_a_vs_iterations(cfg, ["p0_c0p2", "p0_c0p5", "p0_c1p0"],
                                                                basename="fig_p0_stage_a_vs_iterations"),
           "stage_a_p1": lambda cfg: fig_stage_a_vs_iterations(cfg, [v["name"] for v in cfg["ibw"]["variants"]])}


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
