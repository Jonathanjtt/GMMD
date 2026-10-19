#!/usr/bin/env python3
"""Reusable matplotlib defaults and paired publication export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#8A8A8A",
    "light_gray": "#D9D9D9",
}

SERIES_STYLES = (
    (PALETTE["blue"], "-", "o"),
    (PALETTE["vermillion"], "--", "s"),
    (PALETTE["green"], "-.", "^"),
    (PALETTE["orange"], ":", "D"),
    (PALETTE["purple"], (0, (5, 2)), "v"),
    (PALETTE["sky"], (0, (3, 1, 1, 1)), "P"),
)


@dataclass(frozen=True)
class FigureStyle:
    """Physical and typographic defaults for a paper figure."""

    width_in: float = 3.35
    height_in: float = 2.55
    font_size: float = 8.5
    label_size: float = 9.0
    tick_size: float = 8.0
    legend_size: float = 7.5
    line_width: float = 1.6
    marker_size: float = 4.5
    axes_linewidth: float = 0.8
    use_tex: bool = False
    font_family: tuple[str, ...] = (
        "DejaVu Sans",
        "Arial",
        "Helvetica",
        "sans-serif",
    )


def apply_publication_style(style: FigureStyle | None = None) -> FigureStyle:
    """Apply restrained matplotlib defaults and return the resolved style."""

    style = style or FigureStyle()
    mpl.rcParams.update(
        {
            "figure.figsize": (style.width_in, style.height_in),
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.04,
            "font.family": "sans-serif",
            "font.sans-serif": list(style.font_family),
            "font.size": style.font_size,
            "text.usetex": style.use_tex,
            "axes.labelsize": style.label_size,
            "axes.titlesize": style.label_size,
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": style.axes_linewidth,
            "ytick.major.width": style.axes_linewidth,
            "legend.fontsize": style.legend_size,
            "legend.frameon": False,
            "lines.linewidth": style.line_width,
            "lines.markersize": style.marker_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return style


def style_for_series(index: int) -> tuple[str, object, str]:
    """Return a color, line style, and marker with redundant encoding."""

    return SERIES_STYLES[index % len(SERIES_STYLES)]


def add_panel_labels(
    axes: Iterable[plt.Axes],
    labels: Sequence[str] | None = None,
    *,
    x: float = -0.14,
    y: float = 1.04,
) -> None:
    """Add bold panel labels in axes coordinates."""

    axes = list(axes)
    labels = list(labels) if labels is not None else [f"({chr(97 + i)})" for i in range(len(axes))]
    if len(labels) != len(axes):
        raise ValueError("labels and axes must have the same length")
    for ax, label in zip(axes, labels):
        ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", va="bottom")


def save_figure(
    fig: plt.Figure,
    output_basename: str | Path,
    *,
    dpi: int = 400,
    transparent: bool = False,
    close: bool = True,
    tight: bool = False,
    pad_inches: float = 0.04,
) -> tuple[Path, Path]:
    """Save matching PNG and PDF files and return their paths.

    A supplied suffix is removed so that both formats always share one basename.
    """

    if dpi < 300:
        raise ValueError("Publication PNG export requires dpi >= 300")
    base = Path(output_basename).expanduser()
    if base.suffix.lower() in {".png", ".pdf"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")

    common = {
        "bbox_inches": "tight" if tight else None,
        "facecolor": "none" if transparent else "white",
        "transparent": transparent,
    }
    if tight:
        common["pad_inches"] = pad_inches
    fig.savefig(pdf_path, format="pdf", **common)
    fig.savefig(png_path, format="png", dpi=dpi, **common)

    if close:
        plt.close(fig)
    return png_path, pdf_path


__all__ = [
    "FigureStyle",
    "PALETTE",
    "SERIES_STYLES",
    "add_panel_labels",
    "apply_publication_style",
    "save_figure",
    "style_for_series",
]
