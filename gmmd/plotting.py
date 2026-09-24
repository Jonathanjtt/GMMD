"""Figures for GMMD: sample grids and the diagnostic panels.

Functions take already-computed numbers / images and return the matplotlib Figure; nothing is
computed here, and nothing is saved unless the caller asks: :func:`savefig` writes the PNG + PDF
pair.  Keeping the figures out of the experiment driver is what lets a saved run be re-plotted
without touching a GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gmmd.datasets.celebahq import to_uint8

__all__ = ["GROUP_COLORS", "GROUP_LABELS", "tile", "sample_grid_figure", "controls_figure",
           "embedding_figure", "selection_figure", "lrt_figure", "savefig"]

# One colour per sample group in every panel.  The three hues are the first three categorical
# slots of the dataviz reference palette, which validate as a set for scatter (all-pairs) use.
GROUP_COLORS = {"long": "#2a78d6", "near": "#eb6834", "deterministic": "#1baf7a"}
GROUP_LABELS = {"long": r"stochastic $t \to s$", "near": r"stochastic $t \to s_{\rm near}$",
                "deterministic": "probability-flow ODE (repeats)"}


def tile(images, ncols=8, pad=2, fill=255):
    """(n, 3, H, W) float images -> one uint8 (rows*H, cols*W, 3) mosaic with ``pad`` px gaps."""
    imgs = to_uint8(np.asarray(images))
    if imgs.ndim == 3:
        imgs = imgs[None]
    n, h, w, c = imgs.shape
    ncols = int(max(1, min(ncols, n)))
    nrows = int(np.ceil(n / ncols))
    canvas = np.full((nrows * (h + pad) - pad, ncols * (w + pad) - pad, c), fill, np.uint8)
    for k in range(n):
        r, cc = divmod(k, ncols)
        canvas[r * (h + pad):r * (h + pad) + h, cc * (w + pad):cc * (w + pad) + w] = imgs[k]
    return canvas


def _imshow(ax, img, title=None, fontsize=8):
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fontsize)
    return ax


def sample_grid_figure(x0, x_t, x_t_denoised, x_s, x_s_denoised, meta, ncols=8, max_show=64):
    """The H1 figure: ``x_0``, the frozen ``x_t`` (and its one-step denoising), then every
    ``x_s^(i)`` (raw, and Tweedie-denoised for readability) from that SAME ``x_t``."""
    import matplotlib.pyplot as plt

    x_s = np.asarray(x_s)[:max_show]
    x_s_denoised = np.asarray(x_s_denoised)[:max_show]
    n = len(x_s)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(1.55 * ncols + 0.4, 1.55 * nrows * 2 + 3.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.35, nrows, nrows])
    _imshow(fig.add_subplot(gs[0, 0]), to_uint8(x0), f"$x_0$  ({meta.get('image_id', 'image')})")
    _imshow(fig.add_subplot(gs[0, 1]), to_uint8(x_t),
            f"frozen $x_t$,  t = {meta['t']:.4f},  $\\sigma_t$ = {meta['sigma_t']:.3g}\n"
            f"(forward-noise seed {meta.get('forward_noise_seed', '?')})")
    _imshow(fig.add_subplot(gs[0, 2]), to_uint8(x_t_denoised), r"$E_\theta[x_0 \mid x_t]$ (Tweedie, 1 NFE)")
    _imshow(fig.add_subplot(gs[1, :]), tile(x_s, ncols),
            f"{n} independent stochastic rollouts from the SAME $x_t$ to s = {meta['s']:.4f} "
            f"($\\sigma_s$ = {meta['sigma_s']:.3g}); raw $x_s^{{(i)}}$, seeds "
            f"{meta.get('seed_first', '?')}..{meta.get('seed_last', '?')}", fontsize=9)
    _imshow(fig.add_subplot(gs[2, :]), tile(x_s_denoised, ncols),
            r"the same $x_s^{(i)}$, each shown through $E_\theta[x_0 \mid x_s^{(i)}]$ (Tweedie, 1 NFE each)",
            fontsize=9)
    fig.suptitle(f"p_theta(x_s | x_t): {meta.get('sampler', '')}, {meta.get('n_steps', '?')} steps, "
                 f"{meta.get('nfe_per_sample', '?')} NFE per rollout", fontsize=10)
    return fig


def controls_figure(x_near, x_near_denoised, x_det, x_det_denoised, meta, ncols=8, max_show=32):
    """Control 1 (deterministic repeats) and control 2 (the short transition), same layout."""
    import matplotlib.pyplot as plt

    x_near = np.asarray(x_near)[:max_show]
    x_near_denoised = np.asarray(x_near_denoised)[:max_show]
    nrows = int(np.ceil(len(x_near) / ncols))
    fig = plt.figure(figsize=(1.55 * ncols + 0.4, 1.55 * (nrows * 2 + 2) + 2.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, nrows, nrows])
    _imshow(fig.add_subplot(gs[0]), tile(x_det, len(x_det)),
            f"control 1: probability-flow ODE from the same $x_t$ to s = {meta['s']:.4f}, "
            f"{len(x_det)} repeated runs (max |diff| = {meta.get('det_max_abs_diff', float('nan')):.2e})", fontsize=9)
    _imshow(fig.add_subplot(gs[1]), tile(x_det_denoised, len(x_det_denoised)),
            "control 1, Tweedie-denoised", fontsize=9)
    _imshow(fig.add_subplot(gs[2]), tile(x_near, ncols),
            f"control 2: the same rollouts stopped at $s_{{\\rm near}}$ = {meta['s_near']:.4f} "
            f"($\\sigma$ = {meta['sigma_near']:.3g}), raw", fontsize=9)
    _imshow(fig.add_subplot(gs[3]), tile(x_near_denoised, ncols), "control 2, Tweedie-denoised", fontsize=9)
    return fig


def embedding_figure(emb, method, title=None):
    """One 2-D embedding (PCA or UMAP), the groups in fixed colours with a legend."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    for g, E in emb.items():
        E = np.asarray(E)
        if len(E) == 0:
            continue
        ax.scatter(E[:, 0], E[:, 1], s=22, linewidths=0.6, edgecolors="white",
                   color=GROUP_COLORS.get(g), label=f"{GROUP_LABELS.get(g, g)} (n={len(E)})",
                   alpha=0.9, zorder=3 if g == "deterministic" else 2)
    ax.set_xlabel(f"{method} 1"); ax.set_ylabel(f"{method} 2")
    ax.grid(True, color="0.92", linewidth=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    if title:
        ax.set_title(title, fontsize=9)
    return fig


def selection_figure(diags):
    """BIC and held-out log-likelihood against K, one measure per axis, groups in colour."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)
    for g, d in diags.items():
        if d.get("degenerate") or "gmm" not in d:
            continue
        rows = d["gmm"]["rows"]
        K = [r["K"] for r in rows]
        c = GROUP_COLORS.get(g)
        axes[0].plot(K, [r["bic"] - rows[0]["bic"] for r in rows], "-o", color=c, ms=5, lw=1.6,
                     label=GROUP_LABELS.get(g, g))
        axes[1].errorbar(K, [r["cv_loglik_mean"] for r in rows], yerr=[r["cv_loglik_std"] for r in rows],
                         fmt="-o", color=c, ms=5, lw=1.6, capsize=2, label=GROUP_LABELS.get(g, g))
    axes[0].axhline(0, color="0.7", lw=0.8)
    axes[0].set_ylabel("BIC(K) $-$ BIC(1)   (lower is better)")
    axes[1].set_ylabel("held-out log-likelihood per point (higher is better)")
    for ax in axes:
        ax.set_xlabel("mixture components K"); ax.grid(True, color="0.92", linewidth=0.6)
        ax.set_axisbelow(True); ax.xaxis.get_major_locator().set_params(integer=True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=7.5, frameon=False)
    return fig


def lrt_figure(diags):
    """Parametric-bootstrap LRT: the null histogram of 2(LL2-LL1) with the observed statistic."""
    import matplotlib.pyplot as plt

    groups = [g for g, d in diags.items() if not d.get("degenerate") and "lrt" in d]
    fig, axes = plt.subplots(1, max(1, len(groups)), figsize=(4.0 * max(1, len(groups)), 3.2),
                             constrained_layout=True, squeeze=False)
    for ax, g in zip(axes[0], groups):
        l = diags[g]["lrt"]
        ax.hist(l["null"], bins=25, color="0.8", edgecolor="white", label="Gaussian null (bootstrap)")
        ax.axvline(l["statistic"], color=GROUP_COLORS.get(g), lw=2,
                   label=f"observed = {l['statistic']:.1f}\np = {l['p_value']:.3f}")
        ax.set_title(GROUP_LABELS.get(g, g), fontsize=9)
        ax.set_xlabel(r"$2\,(\log L_{K=2} - \log L_{K=1})$"); ax.set_ylabel("count")
        ax.legend(fontsize=7.5, frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    return fig


def savefig(fig, stem, dpi=150):
    """Write ``<stem>.png`` and ``<stem>.pdf``; returns the two paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for p in paths:
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    return paths
