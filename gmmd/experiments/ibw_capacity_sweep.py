"""Experiment 3 continuation: capacity, Monte-Carlo budget, and optimization diagnostics.

This is intentionally a staged experiment.  A cheap screen is run first and later configs can
append only the promising K/B_grad/iteration/step-size combinations.  Completed fits and fresh
teacher rollouts are checkpointed independently, so extending the run never overwrites the first
Experiment 3 or repeats expensive score-network work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from gmmd import config as C
from gmmd import diagnostics as D
from gmmd import features as Fe
from gmmd import ibw
from gmmd import ibw_gpu
from gmmd import provenance as Pv
from gmmd.conditional_target import VEConditionalTarget
from gmmd.experiments.ibw_conditional import teacher_mode_init
from gmmd.experiments.ibw_initialization import _component_stats, _denoise
from gmmd.plotting import savefig

ROOT = Path(__file__).resolve().parents[2]


def expand_run_grid(run_specs):
    """Expand TOML run blocks into deterministic, duplicate-free fit specifications."""
    rows, seen = [], set()
    required = ("k", "bg", "seeds", "iterations", "step_sizes")
    for block in run_specs:
        missing = [x for x in required if x not in block]
        if missing:
            raise ValueError(f"run block is missing {missing}")
        for k in block["k"]:
            for bg in block["bg"]:
                for seed in block["seeds"]:
                    for iterations in block["iterations"]:
                        for step_size in block["step_sizes"]:
                            row = dict(stage=str(block.get("stage", "unspecified")),
                                       init=str(block.get("init", "standard")), k=int(k),
                                       bg=int(bg), seed=int(seed), iterations=int(iterations),
                                       step_size=float(step_size))
                            if row["init"] not in ("standard", "teacher", "spread"):
                                raise ValueError(f"unknown initialization {row['init']!r}")
                            if min(row["k"], row["bg"], row["iterations"]) < 1 or row["step_size"] <= 0:
                                raise ValueError(f"invalid run specification: {row}")
                            key = tuple(row.values())
                            if key in seen:
                                continue
                            seen.add(key); rows.append(row)
    return rows


def _run_id(spec):
    step = f"{spec['step_size']:.8g}".replace(".", "p")
    return (f"{spec['stage']}_{spec['init']}_k{spec['k']}_bg{spec['bg']}_"
            f"it{spec['iterations']}_step{step}_seed{spec['seed']}")


def _write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0])
    extra = sorted({k for row in rows for k in row} - set(fields))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + extra)
        w.writeheader(); w.writerows(rows)


def _teacher_sampler_settings(reference):
    cfg = json.loads((reference / "config.json").read_text())
    tc = cfg["teacher"]
    return dict(predictor=tc["predictor"], corrector=tc["corrector"],
                snr=float(tc["snr"]), n_steps_each=int(tc["n_steps_each"]))


def _ensure_teacher_eval(out, reference, teacher, target, cfg, log):
    """Generate a held-out teacher pool in restartable batches and cache DINOv2 features."""
    pool_path = out / "teacher_eval" / "pool.npz"
    feature_path = out / "teacher_eval" / "features.npz"
    n = int(cfg["n_samples"])
    seeds = np.arange(int(cfg["rollout_seed_base"]), int(cfg["rollout_seed_base"]) + n)
    if pool_path.exists() and feature_path.exists():
        z = np.load(pool_path)
        if np.array_equal(z["rollout_seeds"], seeds):
            log(f"[exp3-cap] reuse {n}-sample held-out teacher pool")
            return z["x_s"], z["x_s_denoised"], np.load(feature_path)["dinov2"]

    batches = out / "teacher_eval" / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    bs = int(cfg["batch_size"])
    sampler = _teacher_sampler_settings(reference)
    source = np.load(reference / "samples.npz")
    raw_parts, den_parts = [], []
    for start in range(0, n, bs):
        batch_seeds = seeds[start:start + bs]
        path = batches / f"seeds_{batch_seeds[0]}_{batch_seeds[-1]}.npz"
        if path.exists():
            z = np.load(path)
            if np.array_equal(z["rollout_seeds"], batch_seeds):
                raw_parts.append(z["x_s"]); den_parts.append(z["x_s_denoised"])
                log(f"[exp3-cap] reuse teacher rollouts {start + 1}-{start + len(batch_seeds)}/{n}")
                continue
        result = teacher.sample_transition(source["x_t"], target.t, target.s,
                                           batch_seeds.tolist(), **sampler)
        raw = result.x_s_np.astype(np.float32)
        den = _denoise(teacher, raw, target.s, bs)
        np.savez_compressed(path, x_s=raw, x_s_denoised=den, rollout_seeds=batch_seeds,
                            t=target.t, s=target.s)
        raw_parts.append(raw); den_parts.append(den)
        log(f"[exp3-cap] teacher rollouts {start + 1}-{start + len(batch_seeds)}/{n}: "
            f"{result.runtime_s:.1f}s")
    raw, den = np.concatenate(raw_parts), np.concatenate(den_parts)
    np.savez_compressed(pool_path, x_s=raw, x_s_denoised=den, rollout_seeds=seeds,
                        t=target.t, s=target.s)
    log(f"[exp3-cap] extracting DINOv2 features for {n} held-out teacher samples")
    features = Fe.extract("dinov2_vitb14", den, device=cfg["feature_device"], batch_size=bs)
    np.savez_compressed(feature_path, dinov2=features, rollout_seeds=seeds)
    return raw, den, features


def _teacher_baseline(features, cfg):
    size = int(cfg["baseline_group_size"])
    if len(features) < 2 * size:
        raise ValueError("teacher_eval.n_samples must be at least twice baseline_group_size")
    rows = []
    for rep in range(int(cfg["baseline_repeats"])):
        idx = np.random.default_rng(80_000 + rep).permutation(len(features))
        a, b = features[idx[:size]], features[idx[size:2 * size]]
        mmd = D.mmd_permutation_test(a, b, n_perm=int(cfg["n_perm"]), seed=90_000 + rep)
        swd = D.sliced_wasserstein(a, b, n_projections=int(cfg["n_projections"]),
                                   seed=90_000 + rep)
        rows.append(dict(repeat=rep, n_per_group=size, mmd2=mmd["mmd2"],
                         mmd_p_value=mmd["p_value"], mmd_null_q95=mmd["null_q95"],
                         mmd_bandwidth=mmd["bandwidth"], sliced_wasserstein=swd["mean"],
                         swd_projection_std=swd["std"]))
    return rows


def _reference_variance(seed, reference_dir):
    run = reference_dir.parent / "runs" / f"k1_standard_seed{seed}.npz"
    if run.exists():
        return float(np.load(run)["variances"].mean()), str(run)
    candidates = sorted((reference_dir.parent / "runs").glob("k1_standard_seed*.npz"))
    if candidates:
        return float(np.mean([np.load(p)["variances"].mean() for p in candidates])), \
               "mean of completed Experiment 3 K=1 fits"
    raise FileNotFoundError("teacher initialization needs a completed Experiment 3 K=1 fit")


def _spread_init(target, k, seed, variance):
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((k, target.dim))
    directions -= directions.mean(0, keepdims=True)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    radius = 1.25 * np.sqrt(target.gap * target.dim)
    return target.x_t[None] + radius * directions, np.full(k, variance)


def _initialization(spec, target, discovery_raw, discovery_den, reference, feature_device,
                    batch_size, log):
    if spec["init"] == "standard":
        means, variances = target.initial_mixture(spec["k"], np.random.default_rng(spec["seed"]))
        return means, variances, "repository prior-free conditional initialization", None
    variance, source = _reference_variance(spec["seed"], reference)
    if spec["init"] == "teacher":
        means, variances = teacher_mode_init(
            discovery_raw, discovery_den, spec["k"], variance, "dinov2_vitb14",
            feature_device, batch_size, 0, log)
        return means, variances, f"oracle DINOv2 teacher-cluster representatives; variance from {source}", source
    means, variances = _spread_init(target, spec["k"], spec["seed"], variance)
    return means, variances, f"teacher-free centered random spread; variance from {source}", source


def _trace_callback(init_means, init_variances, iterations, every, threshold):
    trace = [dict(iteration=0, **_component_stats(init_means, init_variances, threshold))]

    def callback(iteration, q):
        if iteration == iterations or not iteration % every:
            trace.append(dict(iteration=int(iteration), **_component_stats(
                q.means.detach().double().cpu().numpy(),
                q.variances.detach().double().cpu().numpy(), threshold)))
    return trace, callback


def _fit_or_load(spec, run_dir, target, init_means, init_variances, exp, log):
    npz_path = run_dir / f"{_run_id(spec)}.npz"
    if npz_path.exists():
        z = np.load(npz_path)
        fit = {k: z[k] for k in ("means", "variances", "grad_norms", "variances_trace")}
        fit.update(runtime_s=float(z["runtime_s"]), stopped_early=str(z["stopped_early"]) or None,
                   n_iterations_run=int(z["n_iterations_run"]))
        log(f"[exp3-cap] reuse fit {_run_id(spec)}")
        return fit, [], npz_path
    trace, callback = _trace_callback(init_means, init_variances, spec["iterations"],
                                      int(exp["trace_every"]), float(exp["collapse_threshold"]))
    log(f"[exp3-cap] fit {_run_id(spec)}")
    fit = ibw_gpu.fit(target, n_components=spec["k"], method="ibw",
                      n_iterations=spec["iterations"], step_size=spec["step_size"],
                      n_grad_samples=spec["bg"], init_means=init_means,
                      init_variances=init_variances, seed=spec["seed"], device=exp["device"],
                      callback=callback)
    np.savez_compressed(npz_path, means=fit["means"], variances=fit["variances"],
                        init_means=init_means, init_variances=init_variances,
                        grad_norms=fit["grad_norms"], variances_trace=fit["variances_trace"],
                        runtime_s=fit["runtime_s"], stopped_early=fit["stopped_early"] or "",
                        n_iterations_run=fit["n_iterations_run"])
    return fit, trace, npz_path


def _evaluate(fit, target, teacher, teacher_features, spec, exp):
    n = int(exp["n_eval"])
    eval_seed = (100_000 + spec["seed"] + 101 * spec["k"] + 1009 * spec["bg"] +
                 (0 if spec["init"] == "standard" else 10_000))
    q = ibw.IsotropicMixture(fit["means"], fit["variances"])
    raw = q.sample(n, np.random.default_rng(eval_seed)).reshape(-1, *target.image_shape).astype(np.float32)
    den = _denoise(teacher, raw, target.s, int(exp["batch_size"]))
    features = Fe.extract("dinov2_vitb14", den, device=exp["feature_device"],
                          batch_size=int(exp["batch_size"]))
    ref = teacher_features[:n]
    mmd = D.mmd_permutation_test(features, ref, n_perm=int(exp["n_perm"]), seed=eval_seed)
    swd = D.sliced_wasserstein(features, ref, n_projections=int(exp["n_projections"]),
                               seed=eval_seed)
    return dict(n_eval=n, eval_seed=eval_seed, mmd2=mmd["mmd2"],
                mmd_p_value=mmd["p_value"], mmd_null_q95=mmd["null_q95"],
                mmd_bandwidth=mmd["bandwidth"], sliced_wasserstein=swd["mean"],
                swd_projection_std=swd["std"])


def _row(spec, fit, init_means, init_variances, quality, exp):
    sep = _component_stats(fit["means"], fit["variances"], float(exp["collapse_threshold"]))
    init_sep = _component_stats(init_means, init_variances, float(exp["collapse_threshold"]))
    g = np.asarray(fit["grad_norms"], float)
    tail = max(1, len(g) // 10)
    displacement = np.sqrt(np.mean((fit["means"] - init_means) ** 2))
    return dict(**spec, run_id=_run_id(spec),
                initial_normalized_separation_mean=init_sep["normalized_separation_mean"],
                final_variance_mean=float(np.mean(fit["variances"])),
                mean_rms_displacement=float(displacement), grad_norm_first=float(g[0]),
                grad_norm_last=float(g[-1]), grad_norm_tail_mean=float(g[-tail:].mean()),
                grad_norm_tail_std=float(g[-tail:].std()), runtime_s=float(fit["runtime_s"]),
                score_samples=int(spec["k"] * spec["bg"] * fit["n_iterations_run"]),
                network_batches_estimate=int(fit["n_iterations_run"] *
                                             math.ceil(spec["k"] * spec["bg"] /
                                                       int(exp["batch_size"]))),
                stopped_early=fit["stopped_early"], **sep, **quality)


def _aggregate(rows):
    grouped = defaultdict(list)
    keys = ("stage", "init", "k", "bg", "iterations", "step_size")
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, rr in grouped.items():
        item = dict(zip(keys, key)); item["n_seeds"] = len(rr)
        for field in ("mmd2", "sliced_wasserstein", "runtime_s", "grad_norm_tail_mean",
                      "normalized_separation_mean"):
            values = np.asarray([r[field] for r in rr if r[field] is not None], float)
            item[f"{field}_mean"] = float(values.mean()) if len(values) else None
            item[f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out.append(item)
    return sorted(out, key=lambda x: (x["stage"], x["init"], x["bg"], x["iterations"],
                                      x["step_size"], x["k"]))


def _figures(rows, baseline, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = _aggregate(rows)
    baseline_mmd = np.asarray([x["mmd2"] for x in baseline])
    baseline_swd = np.asarray([x["sliced_wasserstein"] for x in baseline])
    styles = {"standard": ("o", "#3976af"), "teacher": ("s", "#d95f45"),
              "spread": ("^", "#55a868")}

    def metric_plot(field, baseline_values, ylabel, stem):
        fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
        groups = defaultdict(list)
        for x in agg:
            if x["stage"] == "screen":
                groups[(x["init"], x["bg"], x["iterations"], x["step_size"])].append(x)
        for ((init, bg, iterations, step), values) in groups.items():
            values.sort(key=lambda x: x["k"])
            marker, color = styles[init]
            ax.errorbar([x["k"] for x in values], [x[f"{field}_mean"] for x in values],
                        yerr=[x[f"{field}_std"] for x in values], marker=marker, color=color,
                        capsize=3, lw=1.6, label=f"{init} capacity screen")
        selected = [r for r in rows if r["init"] == "teacher" and r["k"] == 8 and
                    r["bg"] == 4 and r["iterations"] == 400 and
                    np.isclose(r["step_size"], 0.01)]
        if selected:
            values = np.asarray([r[field] for r in selected], float)
            ax.errorbar([8], [values.mean()], yerr=[values.std(ddof=1) if len(values) > 1 else 0],
                        fmt="D", color="black", capsize=4, ms=6,
                        label=f"selected 400-step fit ({len(values)} seeds)")
        mean, std = baseline_values.mean(), baseline_values.std(ddof=1)
        ax.axhspan(mean - std, mean + std, color="0.75", alpha=0.3,
                   label="teacher–teacher mean ± SD")
        ax.axhline(mean, color="0.35", ls="--", lw=1)
        ax.set_xscale("log", base=2); ax.set_xticks(sorted({r["k"] for r in rows}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("mixture components K"); ax.set_ylabel(ylabel)
        ax.set_title("Capacity screen and selected early-stopped fit")
        ax.grid(True, color="0.92", lw=0.7); ax.legend(frameon=False, fontsize=7)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        savefig(fig, out / "figures" / stem); plt.close(fig)

    metric_plot("mmd2", baseline_mmd, r"DINOv2 MMD$^2$ (lower is better)",
                "figure_3e_mmd_capacity")
    metric_plot("sliced_wasserstein", baseline_swd, "DINOv2 sliced Wasserstein",
                "figure_3f_swd_capacity")

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for init in sorted({x["init"] for x in agg}):
        xx = [x for x in agg if x["init"] == init]
        marker, color = styles[init]
        ax.scatter([x["score_samples"] if "score_samples" in x else
                    x["k"] * x["bg"] * x["iterations"] for x in xx],
                   [x["runtime_s_mean"] for x in xx], marker=marker, color=color,
                   s=45, label=init)
    ax.set_xlabel("score-network sample evaluations (K × BG × iterations)")
    ax.set_ylabel("fit runtime (seconds)"); ax.set_title("Fit quality budget and wall-clock cost")
    ax.grid(True, color="0.92", lw=0.7); ax.legend(frameon=False)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    savefig(fig, out / "figures" / "figure_3g_compute_cost"); plt.close(fig)

    # Targeted optimization-path diagnostic at the best screen capacity.
    path_rows = [r for r in rows if r["init"] == "teacher" and r["k"] == 8 and
                 r["bg"] == 4 and r["seed"] == 11 and
                 r["stage"] in ("screen", "optimization", "early_stopping")]
    if path_rows:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
        for step, marker, color in ((0.1, "o", "#3976af"), (0.03, "s", "#d95f45"),
                                    (0.01, "^", "#55a868")):
            rr = sorted((r for r in path_rows if np.isclose(r["step_size"], step)),
                        key=lambda r: r["iterations"])
            if not rr: continue
            horizon = [r["iterations"] * r["step_size"] for r in rr]
            for ax, field in zip(axes, ("mmd2", "sliced_wasserstein")):
                ax.plot(horizon, [r[field] for r in rr], "-", marker=marker, color=color,
                        lw=1.8, ms=6, label=f"step={step:g}")
                for x, r in zip(horizon, rr):
                    ax.annotate(str(r["iterations"]), (x, r[field]), xytext=(3, 4),
                                textcoords="offset points", fontsize=7, color=color)
        selected = [r for r in rows if r["init"] == "teacher" and r["k"] == 8 and
                    r["bg"] == 4 and r["iterations"] == 400 and
                    np.isclose(r["step_size"], 0.01)]
        if len(selected) > 1:
            for ax, field in zip(axes, ("mmd2", "sliced_wasserstein")):
                values = np.asarray([r[field] for r in selected], float)
                ax.errorbar([4.0], [values.mean()], yerr=[values.std(ddof=1)], fmt="D",
                            color="black", ms=5, capsize=4, zorder=5,
                            label=f"400-step mean ± SD ({len(values)} seeds)")
        bm = np.asarray([r["mmd2"] for r in baseline])
        bs = np.asarray([r["sliced_wasserstein"] for r in baseline])
        for ax, vals, ylabel in ((axes[0], bm, r"DINOv2 MMD$^2$"),
                                 (axes[1], bs, "DINOv2 sliced Wasserstein")):
            ax.axhspan(vals.mean() - vals.std(ddof=1), vals.mean() + vals.std(ddof=1),
                       color="0.75", alpha=0.3, label="teacher–teacher mean ± SD")
            ax.set_xlabel("nominal optimization horizon (iterations × step)")
            ax.set_ylabel(ylabel); ax.grid(True, color="0.92", lw=0.7)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        axes[0].set_title("K=8 optimization path (numbers are iterations)")
        axes[1].set_title("Smaller steps and early stopping")
        axes[1].legend(frameon=False, fontsize=7)
        savefig(fig, out / "figures" / "figure_3h_optimization_path"); plt.close(fig)

    # Fair cost/variance view for the selected K=4 BG sweep.
    bg_rows = [r for r in rows if r["init"] == "teacher" and r["k"] == 4 and
               r["iterations"] == 300 and np.isclose(r["step_size"], 0.1) and
               r["stage"] in ("screen", "seed_repeat", "bg_sweep")]
    if bg_rows:
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), constrained_layout=True)
        for ax, field, ylabel in ((axes[0], "mmd2", r"DINOv2 MMD$^2$"),
                                  (axes[1], "sliced_wasserstein", "DINOv2 sliced Wasserstein")):
            xs, means, sds = [], [], []
            for bg in sorted({r["bg"] for r in bg_rows}):
                vals = np.asarray([r[field] for r in bg_rows if r["bg"] == bg], float)
                xs.append(bg); means.append(vals.mean()); sds.append(vals.std(ddof=1) if len(vals)>1 else 0)
            ax.errorbar(xs, means, yerr=sds, color="#6f4c9b", marker="o", capsize=4, lw=2)
            ax.set_xscale("log", base=2); ax.set_xticks(xs)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("gradient samples BG"); ax.set_ylabel(ylabel)
            ax.grid(True, color="0.92", lw=0.7)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        axes[0].set_title("K=4 BG sweep (BG=16: mean ± SD, 3 seeds)")
        axes[1].set_title("More gradient samples increase cost, not accuracy")
        savefig(fig, out / "figures" / "figure_3i_bg_sweep"); plt.close(fig)


def _write_summary(rows, baseline, out):
    agg = _aggregate(rows)
    bm = np.asarray([x["mmd2"] for x in baseline])
    bs = np.asarray([x["sliced_wasserstein"] for x in baseline])
    best = min(agg, key=lambda x: x["mmd2_mean"])
    lines = ["# Experiment 3 continuation — IBW capacity and optimization", "",
             f"Held-out evaluation uses {rows[0]['n_eval']} fresh teacher samples per fit. "
             f"Teacher–teacher baseline (repeated balanced splits): MMD² {bm.mean():.5f} ± "
             f"{bm.std(ddof=1):.5f}; SWD {bs.mean():.5f} ± {bs.std(ddof=1):.5f}.", "",
             "| stage | init | K | BG | iterations | step | seeds | MMD² mean ± SD | SWD mean ± SD | runtime mean (s) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in agg:
        lines.append(f"| {x['stage']} | {x['init']} | {x['k']} | {x['bg']} | "
                     f"{x['iterations']} | {x['step_size']:g} | {x['n_seeds']} | "
                     f"{x['mmd2_mean']:.5f} ± {x['mmd2_std']:.5f} | "
                     f"{x['sliced_wasserstein_mean']:.5f} ± {x['sliced_wasserstein_std']:.5f} | "
                     f"{x['runtime_s_mean']:.1f} |")
    lines += ["", f"Best completed setting: {best['init']} initialization, K={best['k']}, "
              f"BG={best['bg']}, {best['iterations']} iterations, step={best['step_size']:g}; "
              f"MMD²={best['mmd2_mean']:.5f}.",
              ("It reaches the requested MMD² < 0.01 threshold."
               if best["mmd2_mean"] < 0.01 else
               "It remains above MMD² 0.01; follow-up budget and trajectory comparisons are required."),
              "", "## Diagnostic conclusions", ""]
    standard = [r for r in rows if r["stage"] == "screen" and r["init"] == "standard"]
    teacher = [r for r in rows if r["stage"] == "screen" and r["init"] == "teacher"]
    if standard:
        a = next(r for r in standard if r["k"] == 1); b = next(r for r in standard if r["k"] == 16)
        lines.append(f"- Standard initialization barely benefits from 16× capacity: MMD² "
                     f"{a['mmd2']:.5f} → {b['mmd2']:.5f}, while fit runtime rises "
                     f"{a['runtime_s']:.1f}s → {b['runtime_s']:.1f}s.")
    if teacher:
        k1 = next(r for r in teacher if r["k"] == 1)
        kb = min(teacher, key=lambda r: r["mmd2"])
        lines.append(f"- Oracle teacher-cluster initialization is the largest improvement: K=1 "
                     f"MMD² {k1['mmd2']:.5f}; best screen K={kb['k']} gives {kb['mmd2']:.5f}. "
                     "Capacity then saturates and K=16 worsens.")
    bg16 = [r for r in rows if r["stage"] == "seed_repeat"]
    bg64 = [r for r in rows if r["stage"] == "bg_sweep"]
    if bg16 and bg64:
        lines.append(f"- More gradient samples do not help K=4: BG=16 gives MMD² "
                     f"{np.mean([r['mmd2'] for r in bg16]):.5f} ± "
                     f"{np.std([r['mmd2'] for r in bg16], ddof=1):.5f}; BG=64 gives "
                     f"{bg64[0]['mmd2']:.5f}, at much higher runtime.")
    opt = [r for r in rows if r["stage"] in ("optimization", "early_stopping") and
           r["k"] == 8 and r["bg"] == 4 and r["seed"] == 11]
    if opt:
        ob = min(opt, key=lambda r: r["mmd2"])
        lines.append(f"- The best optimization-path point so far is {ob['iterations']} iterations "
                     f"at step {ob['step_size']:g} (MMD² {ob['mmd2']:.5f}). Longer/larger-step "
                     "optimization can worsen teacher-sample MMD despite lower-noise gradients.")
        same_step = [r for r in opt if r["step_size"] == ob["step_size"]]
        longest = max(same_step, key=lambda r: r["iterations"])
        if longest["iterations"] > ob["iterations"]:
            lines.append(f"- Along the common-seed step={ob['step_size']:g} path, mean component "
                         f"separation contracts from {ob['pairwise_distance_mean']:.1f} at the "
                         f"best point to {longest['pairwise_distance_mean']:.1f} at "
                         f"{longest['iterations']} iterations, while MMD² rises to "
                         f"{longest['mmd2']:.5f}. This is objective-driven over-contraction, not "
                         "component collapse or failure to optimize.")
    selected = [r for r in rows if r["init"] == "teacher" and r["k"] == 8 and
                r["bg"] == 4 and r["iterations"] == 400 and
                np.isclose(r["step_size"], 0.01)]
    if len(selected) > 1:
        mmd = np.asarray([r["mmd2"] for r in selected], float)
        swd = np.asarray([r["sliced_wasserstein"] for r in selected], float)
        lines.append(f"- Repeating the selected 400-step setting across {len(selected)} seeds gives "
                     f"MMD² {mmd.mean():.5f} ± {mmd.std(ddof=1):.5f} and SWD "
                     f"{swd.mean():.5f} ± {swd.std(ddof=1):.5f}.")
    lines += ["- The remaining gap is most consistent with target/objective mismatch and limited "
              "mixture-family capacity: the implementation fixes equal component weights and "
              "isotropic pixel-space covariance. More K or Monte-Carlo gradient samples cannot "
              "repair either restriction.", "", "## Prioritized next steps (no conditional network)", "",
              "1. Turn the held-out teacher metric into a validation-only early-stopping rule; "
              "confirm the selected iteration on independent evaluation seeds and report its "
              "mean ± SD rather than selecting on the test split.",
              "2. Run a weighted-mixture control initialized from teacher clusters. The discovery "
              "clusters are strongly imbalanced, while current IBW fixes every component weight "
              "to 1/K; this cleanly tests whether weight capacity is the dominant residual gap.",
              "3. Separate objective from family mismatch with a direct teacher-sample fit/control, "
              "then compare it to score-based IBW under the same K, samples, and evaluation seeds.",
              "4. If weights are insufficient, add a memory-feasible covariance extension "
              "(channelwise/diagonal or low-rank-plus-isotropic), starting at K=4 or K=8. Full "
              "pixel covariance is infeasible at 256² resolution.",
              "5. Only after those controls, revisit step schedules or larger K/BG. Stop increasing "
              "BG now: the measured cost scales nearly linearly without an accuracy gain.", "",
              "All fit checkpoints, convergence traces, resolved settings, and per-run metrics are retained."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def run(cfg, out_dir=None, device=None, log=print):
    started = time.time()
    out = Path(out_dir or ROOT / cfg["output"]["dir"])
    for sub in ("configs", "numerical", "runs", "figures", "teacher_eval"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    Pv.dump_json(cfg, out / "configs" / "resolved_config.json")
    specs = expand_run_grid(cfg["runs"])
    Pv.dump_json(specs, out / "configs" / "expanded_runs.json")

    reference = ROOT / cfg["reference"]["dir"]
    if not (reference / "samples.npz").exists():
        raise FileNotFoundError(f"completed Experiment 3 reference is required: {reference}")
    from gmmd.experiments.ibw_initialization import _build_teacher
    dev = device or cfg["experiment"]["device"]
    teacher = _build_teacher(reference, dev)
    source = np.load(reference / "samples.npz")
    target = VEConditionalTarget(teacher, source["x_t"], float(source["t"]), float(source["s"]),
                                 batch_size=int(cfg["experiment"]["batch_size"]))
    exp = dict(cfg["experiment"]); exp["device"] = dev
    exp["feature_device"] = dev if exp["feature_device"] == "same" else exp["feature_device"]

    teacher_raw, _, teacher_features = _ensure_teacher_eval(
        out, reference, teacher, target, {**cfg["teacher_eval"],
        "feature_device": exp["feature_device"], "n_perm": exp["n_perm"],
        "n_projections": exp["n_projections"]}, log)
    baseline = _teacher_baseline(teacher_features, {**cfg["teacher_eval"],
                                 "n_perm": exp["n_perm"], "n_projections": exp["n_projections"]})
    Pv.dump_json(baseline, out / "numerical" / "teacher_baseline.json")

    n_discovery = int(cfg["reference"]["discovery_samples"])
    discovery_raw = source["x_s"][:n_discovery]
    discovery_den = source["x_s_denoised"][:n_discovery]
    rows = []
    for spec in specs:
        json_path = out / "runs" / f"{_run_id(spec)}.json"
        if json_path.exists():
            rows.append(json.loads(json_path.read_text())["row"])
            log(f"[exp3-cap] reuse completed {_run_id(spec)}")
            continue
        init_means, init_vars, init_note, variance_source = _initialization(
            spec, target, discovery_raw, discovery_den, reference, exp["feature_device"],
            int(exp["batch_size"]), log)
        fit, trajectory, npz_path = _fit_or_load(spec, out / "runs", target,
                                                 init_means, init_vars, exp, log)
        quality = _evaluate(fit, target, teacher, teacher_features, spec, exp)
        row = _row(spec, fit, init_means, init_vars, quality, exp)
        if not trajectory:
            trajectory = [dict(iteration=spec["iterations"], **_component_stats(
                fit["means"], fit["variances"], float(exp["collapse_threshold"])))]
        Pv.dump_json(dict(row=row, trajectory=trajectory, init_note=init_note,
                          variance_source=variance_source, fit_checkpoint=str(npz_path)), json_path)
        rows.append(row)
        log(f"[exp3-cap] {row['run_id']}: MMD²={row['mmd2']:.5f}, "
            f"SWD={row['sliced_wasserstein']:.5f}, runtime={row['runtime_s']:.1f}s")

    rows.sort(key=lambda x: specs.index({k: x[k] for k in
              ("stage", "init", "k", "bg", "seed", "iterations", "step_size")}))
    _write_csv(rows, out / "numerical" / "run_metrics.csv")
    aggregates = _aggregate(rows)
    Pv.dump_json(dict(rows=rows, aggregates=aggregates, teacher_baseline=baseline,
                      target=target.describe()), out / "numerical" / "results.json")
    _figures(rows, baseline, out); _write_summary(rows, baseline, out)
    provenance = Pv.snapshot(cfg, dev)
    provenance.update(status="finished", runtime_s=time.time() - started,
                      reference=str(reference), target=target.describe(), n_runs=len(rows),
                      heldout_teacher_samples=len(teacher_raw))
    Pv.dump_json(provenance, out / "provenance.json")
    log(f"[exp3-cap] complete -> {out}")
    return rows


def replot(cfg, out_dir=None):
    out = Path(out_dir or ROOT / cfg["output"]["dir"])
    payload = json.loads((out / "numerical" / "results.json").read_text())
    _figures(payload["rows"], payload["teacher_baseline"], out)
    _write_summary(payload["rows"], payload["teacher_baseline"], out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="exp3_ibw_capacity")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args(argv)
    cfg = C.load_config(args.config)
    return replot(cfg, args.out) if args.plot_only else run(cfg, args.out, args.device)


if __name__ == "__main__":
    main()
