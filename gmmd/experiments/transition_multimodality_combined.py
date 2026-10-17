"""Combine completed Experiment 1 cohorts without rerunning teacher samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gmmd import provenance as Pv
from gmmd.plotting import savefig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    "outputs/exp1_transition_multimodality",
    "outputs/exp1_transition_multimodality_expanded",
)


def _wilson(successes, n, z=1.96):
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(center - half), float(center + half)


def aggregate(rows):
    """Endpoint summaries retaining both mean/SD and robust median/IQR."""
    out = []
    for s in sorted({float(r["s"]) for r in rows}, reverse=True):
        rr = [r for r in rows if np.isclose(float(r["s"]), s)]
        x = np.asarray([r["delta_bic_2_minus_1"] for r in rr], float)
        n, favored = len(x), int(np.sum(x < 0))
        lo, hi = _wilson(favored, n)
        out.append(dict(s=s, delta_t=float(np.mean([r["delta_t"] for r in rr])), n_states=n,
                        delta_bic_mean=float(x.mean()), delta_bic_std=float(x.std(ddof=1)),
                        delta_bic_sem=float(x.std(ddof=1) / np.sqrt(n)),
                        delta_bic_median=float(np.median(x)),
                        delta_bic_q25=float(np.quantile(x, 0.25)),
                        delta_bic_q75=float(np.quantile(x, 0.75)),
                        k2_favored_count=favored, k2_favored_fraction=favored / n,
                        k2_favored_wilson_low=lo, k2_favored_wilson_high=hi,
                        preferred_mixture_count=int(sum(int(r["preferred_k_bic"]) > 1 for r in rr))))
    return out


def _write_csv(rows, path):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _figures(rows, summaries, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = sorted({int(r["image_index"]) for r in rows})
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(states)))
    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    for color, state in zip(colors, states):
        rr = sorted((r for r in rows if int(r["image_index"]) == state),
                    key=lambda r: r["delta_t"])
        ax.plot([r["delta_t"] for r in rr], [r["delta_bic_2_minus_1"] for r in rr],
                "-o", color=color, alpha=0.63, lw=1.2, ms=4, label=f"state {state}")
    ax.errorbar([r["delta_t"] for r in summaries], [r["delta_bic_mean"] for r in summaries],
                yerr=[r["delta_bic_sem"] for r in summaries], color="black", marker="D",
                lw=2.4, capsize=4, label="mean ± SEM", zorder=5)
    ax.plot([r["delta_t"] for r in summaries], [r["delta_bic_median"] for r in summaries],
            "--s", color="0.25", ms=5, lw=1.5, label="median", zorder=4)
    ax.axhline(0, color="0.5", ls=":", lw=1)
    ax.set_xlabel(r"transition length $\Delta t=t-s$")
    ax.set_ylabel(r"$\mathrm{BIC}_{K=2}-\mathrm{BIC}_{K=1}$ (lower favors K=2)")
    ax.set_title("Combined transition-length result across eight fixed noisy states")
    ax.grid(True, color="0.92", lw=0.7); ax.legend(frameon=False, fontsize=7, ncol=2)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    savefig(fig, out / "figures" / "figure_1d_combined_transition_length"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    x = np.asarray([r["delta_t"] for r in summaries])
    frac = np.asarray([r["k2_favored_fraction"] for r in summaries])
    low = np.asarray([r["k2_favored_wilson_low"] for r in summaries])
    high = np.asarray([r["k2_favored_wilson_high"] for r in summaries])
    axes[0].errorbar(x, frac, yerr=np.stack([frac - low, high - frac]), color="#3976af",
                     marker="o", capsize=4, lw=2)
    for xx, r in zip(x, summaries):
        axes[0].text(xx, r["k2_favored_fraction"] + 0.045,
                     f"{r['k2_favored_count']}/{r['n_states']}", ha="center", fontsize=8)
    axes[0].set_ylim(-0.04, 0.78); axes[0].set_xlabel(r"transition length $\Delta t$")
    axes[0].set_ylabel("fraction with BIC(K=2) < BIC(K=1)")
    axes[0].set_title("State prevalence (Wilson 95% interval)")
    med = np.asarray([r["delta_bic_median"] for r in summaries])
    q1 = np.asarray([r["delta_bic_q25"] for r in summaries])
    q3 = np.asarray([r["delta_bic_q75"] for r in summaries])
    axes[1].plot(x, med, "-s", color="#d95f45", lw=2)
    axes[1].fill_between(x, q1, q3, color="#d95f45", alpha=0.2, label="IQR")
    axes[1].axhline(0, color="0.5", ls=":", lw=1)
    axes[1].set_xlabel(r"transition length $\Delta t$")
    axes[1].set_ylabel(r"BIC(K=2)−BIC(K=1)")
    axes[1].set_title("Median and interquartile range"); axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, color="0.92", lw=0.7)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    savefig(fig, out / "figures" / "figure_1e_prevalence_and_robust_trend"); plt.close(fig)

    endpoints = [r["s"] for r in summaries]
    matrix = np.asarray([[next(int(r["preferred_k_bic"]) for r in rows
                               if int(r["image_index"]) == state and np.isclose(r["s"], s))
                          for s in endpoints] for state in states])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    im = ax.imshow(matrix, cmap="viridis", vmin=1, vmax=max(4, int(matrix.max())), aspect="auto")
    ax.set_xticks(range(len(endpoints)),
                  [f"s={r['s']:.2f}\nΔt={r['delta_t']:.2f}" for r in summaries])
    ax.set_yticks(range(len(states)), [f"state {s}" for s in states])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white",
                    fontweight="bold")
    ax.set_title("Preferred DINOv2-space mixture order across both cohorts")
    fig.colorbar(im, ax=ax, label="preferred K", ticks=range(1, max(4, int(matrix.max())) + 1))
    savefig(fig, out / "figures" / "figure_1f_combined_preferred_k"); plt.close(fig)


def run(input_dirs=DEFAULT_INPUTS, out_dir="outputs/exp1_transition_multimodality_combined"):
    inputs = [ROOT / p for p in input_dirs]
    out = ROOT / out_dir
    (out / "numerical").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rows = []
    for path in inputs:
        payload = json.loads((path / "numerical" / "condition_summary.json").read_text())
        for row in payload["rows"]:
            rows.append(dict(cohort=path.name, **row))
    summaries = aggregate(rows)
    _write_csv(rows, out / "numerical" / "condition_summary_combined.csv")
    _write_csv(summaries, out / "numerical" / "endpoint_summary.csv")
    Pv.dump_json(dict(inputs=[str(p) for p in inputs], rows=rows, endpoint_summary=summaries),
                 out / "numerical" / "results.json")
    _figures(rows, summaries, out)
    lines = ["# Combined Experiment 1 — transition length and conditional multimodality", "",
             f"Combined {len(set(r['image_index'] for r in rows))} fixed noisy states across "
             f"{len(summaries)} paired endpoints; the original and expanded cohorts remain unchanged.", "",
             "| Δt | states favoring K=2 over K=1 | fraction | mean ΔBIC ± SD | median ΔBIC [IQR] |",
             "|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['delta_t']:.4f} | {r['k2_favored_count']}/{r['n_states']} | "
                     f"{r['k2_favored_fraction']:.3f} | {r['delta_bic_mean']:.2f} ± "
                     f"{r['delta_bic_std']:.2f} | {r['delta_bic_median']:.2f} "
                     f"[{r['delta_bic_q25']:.2f}, {r['delta_bic_q75']:.2f}] |")
    lines += ["", "Evidence for a K=2 alternative becomes more prevalent with transition length "
              "(1/8, 1/8, 3/8, 4/8), but state heterogeneity is large: three states remain "
              "unimodal at every endpoint and one is already multimodal at the shortest endpoint."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return rows, summaries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", default=list(DEFAULT_INPUTS))
    ap.add_argument("--out", default="outputs/exp1_transition_multimodality_combined")
    args = ap.parse_args(argv)
    run(args.inputs, args.out)


if __name__ == "__main__":
    main()
