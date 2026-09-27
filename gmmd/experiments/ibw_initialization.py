"""Experiment 3: diagnose IBW mode discovery versus mixture capacity.

The canonical long H1 transition is regenerated under this experiment's own output directory
when the clone does not contain its gitignored raw arrays.  Five fit conditions then vary only K
and initial component placement.  Teacher-cluster initialisation is explicitly an oracle control.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from gmmd import config as C
from gmmd import diagnostics as D
from gmmd import features as Fe
from gmmd import ibw
from gmmd import ibw_gpu
from gmmd import provenance as Pv
from gmmd.conditional_target import VEConditionalTarget
from gmmd.plotting import savefig

ROOT = Path(__file__).resolve().parents[2]

CONDITIONS = (
    ("k1_standard", 1, "standard", "K=1, standard"),
    ("k2_standard", 2, "standard", "K=2, standard"),
    ("k4_standard", 4, "standard", "K=4, standard"),
    ("k2_spread", 2, "spread", "K=2, deliberately spread"),
    ("k2_teacher", 2, "teacher", "K=2, teacher-cluster oracle"),
)


def _component_stats(means, variances, collapse_threshold):
    means, variances = np.asarray(means), np.asarray(variances)
    k = len(means)
    if k < 2:
        return dict(pairwise_distance_min=None, pairwise_distance_mean=None,
                    normalized_separation_min=None, normalized_separation_mean=None,
                    collapsed=None, effective_separated_components=1)
    raw, norm = [], []
    for i in range(k):
        for j in range(i + 1, k):
            dist = float(np.linalg.norm(means[i] - means[j]))
            raw.append(dist)
            norm.append(dist / (np.sqrt(variances[i]) + np.sqrt(variances[j])))
    # Components linked by a sub-threshold pair are treated as one occupied geometric group.
    parent = list(range(k))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    p = 0
    for i in range(k):
        for j in range(i + 1, k):
            if norm[p] < collapse_threshold: union(i, j)
            p += 1
    return dict(pairwise_distance_min=float(np.min(raw)), pairwise_distance_mean=float(np.mean(raw)),
                normalized_separation_min=float(np.min(norm)),
                normalized_separation_mean=float(np.mean(norm)),
                collapsed=bool(np.min(norm) < collapse_threshold),
                effective_separated_components=len({find(i) for i in range(k)}))


def _trajectory_callback(init_means, init_variances, n_iterations, every, threshold):
    trace = []
    initial = _component_stats(init_means, init_variances, threshold)
    trace.append(dict(iteration=0, **initial))

    def callback(iteration, q):
        if iteration != n_iterations and iteration % every:
            return
        means = q.means.detach().double().cpu().numpy()
        variances = q.variances.detach().double().cpu().numpy()
        trace.append(dict(iteration=int(iteration), **_component_stats(means, variances, threshold)))
    return trace, callback


def _ensure_reference(out, cfg, device, log):
    reference = out / "reference_h1"
    if (reference / "samples.npz").exists() and (reference / "features.npz").exists():
        log(f"[exp3] reusing complete reference at {reference}")
        return reference
    from gmmd.experiments.conditional_multimodality import run as h1_run

    rcfg = C.load_config(cfg["reference"]["base_config"], {
        "features": {"extractor": "dinov2_vitb14", "input": "denoised"},
        "diagnostics": {"n_boot": int(cfg["reference"]["n_boot"]),
                        "n_null": int(cfg["reference"]["n_null"])},
        "output": {"dir": "unused", "save_samples": True},
    })
    rcfg["name"] = "exp3_reference_h1"
    log("[exp3] raw canonical H1 arrays are absent; regenerating them inside exp3 output")
    h1_run(rcfg, device=device, out_dir=reference, log=log)
    return reference


def _build_teacher(reference, device):
    from gmmd.experiments.ibw_conditional import _build_teacher
    return _build_teacher(reference, device)[0]


def _teacher_representatives(features, raw, seed):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=20, random_state=seed).fit(features)
    indices = [int(np.argmin(np.linalg.norm(features - center, axis=1)))
               for center in km.cluster_centers_]
    return raw[indices].reshape(2, -1).astype(np.float64), indices, np.bincount(km.labels_, minlength=2)


def _spread_init(target, seed, variance, radius_scale):
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(target.dim)
    direction /= np.linalg.norm(direction)
    radius = float(radius_scale) * np.sqrt(target.gap * target.dim)
    means = np.stack([target.x_t - radius * direction, target.x_t + radius * direction])
    return means, np.full(2, variance)


def _denoise(teacher, images, s, batch_size):
    import torch
    return np.concatenate([
        teacher.tweedie_denoise(torch.as_tensor(images[i:i + batch_size]), s).cpu().numpy()
        for i in range(0, len(images), batch_size)
    ]).astype(np.float32)


def _evaluate(means, variances, teacher, target, teacher_features, cfg, seed):
    q = ibw.IsotropicMixture(means, variances)
    n_eval = int(cfg["n_eval"])
    sample = q.sample(n_eval, np.random.default_rng(seed + 50_000))
    images = sample.reshape(-1, *target.image_shape).astype(np.float32)
    denoised = _denoise(teacher, images, target.s, int(cfg["batch_size"]))
    features = Fe.extract("dinov2_vitb14", denoised, device=cfg["feature_device"],
                          batch_size=int(cfg["batch_size"]))
    mmd = D.mmd_permutation_test(features, teacher_features, n_perm=int(cfg["n_perm"]), seed=seed)
    swd = D.sliced_wasserstein(features, teacher_features,
                               n_projections=int(cfg["n_projections"]), seed=seed)
    return dict(mmd2=mmd["mmd2"], mmd_p_value=mmd["p_value"],
                mmd_null_q95=mmd["null_q95"], mmd_bandwidth=mmd["bandwidth"],
                sliced_wasserstein=swd["mean"], swd_projection_std=swd["std"])


def _write_csv(rows, path):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _representative_seed(rows, condition):
    rr = [r for r in rows if r["condition"] == condition]
    mean = np.mean([r["mmd2"] for r in rr])
    return min(rr, key=lambda r: abs(r["mmd2"] - mean))["seed"]


def _figures(rows, trajectories, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [x[0] for x in CONDITIONS]
    labels = {x[0]: x[3] for x in CONDITIONS}
    colors = dict(zip(order, plt.cm.tab10(np.linspace(0, 0.8, len(order)))))
    xpos = np.arange(len(order))

    def points_with_mean(ax, field, ylabel, include_k1=True):
        rng = np.random.default_rng(9)
        for i, name in enumerate(order):
            vals = np.asarray([r[field] for r in rows if r["condition"] == name], float)
            vals = vals[np.isfinite(vals)]
            if not len(vals) or (not include_k1 and name == "k1_standard"): continue
            jitter = rng.uniform(-0.09, 0.09, len(vals))
            ax.scatter(i + jitter, vals, color=colors[name], s=30, alpha=0.75, zorder=2)
            ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals)>1 else 0,
                        fmt="D", color="black", ms=5, capsize=3, zorder=3)
        ax.set_xticks(xpos, [labels[x] for x in order], rotation=22, ha="right")
        ax.set_ylabel(ylabel); ax.grid(True, axis="y", color="0.92", lw=0.7)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    points_with_mean(ax, "mmd2", r"DINOv2 MMD$^2$ vs teacher (lower is better)")
    ax.set_title("IBW fit quality by component count and initialisation")
    savefig(fig, out / "figures" / "figure_3a_fit_quality")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    points_with_mean(ax, "normalized_separation_mean",
                     r"mean pairwise separation / $(\sigma_i+\sigma_j)$", include_k1=False)
    ax.axhline(1.0, color="0.45", ls="--", lw=1, label="collapse threshold")
    ax.legend(frameon=False); ax.set_title("Final component separation")
    savefig(fig, out / "figures" / "figure_3b_component_separation")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    for name in order[1:]:
        rr = [r for r in rows if r["condition"] == name]
        ax.scatter([r["normalized_separation_mean"] for r in rr], [r["mmd2"] for r in rr],
                   s=45, color=colors[name], label=labels[name], alpha=0.8)
    ax.set_xlabel(r"mean pairwise separation / $(\sigma_i+\sigma_j)$")
    ax.set_ylabel(r"DINOv2 MMD$^2$ vs teacher (lower is better)")
    ax.set_title("Does component separation predict fit quality?")
    ax.grid(True, color="0.92", lw=0.7); ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    savefig(fig, out / "figures" / "figure_3c_quality_vs_separation")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for name in ("k2_standard", "k2_teacher"):
        seed = _representative_seed(rows, name)
        tr = trajectories[(name, seed)]
        ax.plot([x["iteration"] for x in tr], [x["normalized_separation_mean"] for x in tr],
                lw=2, color=colors[name], label=f"{labels[name]} (seed {seed})")
    ax.axhline(1.0, color="0.45", ls="--", lw=1, label="collapse threshold")
    ax.set_xlabel("IBW iteration"); ax.set_ylabel(r"component separation / $(\sigma_1+\sigma_2)$")
    ax.set_title("Representative component-separation trajectories")
    ax.set_yscale("log"); ax.grid(True, color="0.92", lw=0.7); ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    savefig(fig, out / "figures" / "figure_3d_component_evolution")
    plt.close(fig)


def run(cfg, out_dir=None, device=None, log=print):
    started = time.time()
    out = Path(out_dir or ROOT / cfg["output"]["dir"])
    for sub in ("configs", "numerical", "runs", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    Pv.dump_json(cfg, out / "configs" / "resolved_config.json")
    dev = device or cfg["experiment"]["device"]
    reference = _ensure_reference(out, cfg, dev, log)
    samples = np.load(reference / "samples.npz")
    features = np.load(reference / "features.npz")
    teacher_features = features["dinov2_vitb14/denoised/long"]
    teacher_raw, teacher_den = samples["x_s"], samples["x_s_denoised"]
    teacher = _build_teacher(reference, dev)
    target = VEConditionalTarget(teacher, samples["x_t"], float(samples["t"]),
                                 float(samples["s"]), batch_size=int(cfg["experiment"]["batch_size"]))
    exp = dict(cfg["experiment"])
    exp["feature_device"] = dev if exp["feature_device"] == "same" else exp["feature_device"]
    oracle_means, oracle_indices, oracle_sizes = _teacher_representatives(
        teacher_features, teacher_raw, int(exp["teacher_cluster_seed"]))
    log(f"[exp3] oracle representative samples {oracle_indices}, cluster sizes {oracle_sizes.tolist()}")

    rows, trajectories, baseline_var = [], {}, {}
    seeds = [int(v) for v in exp["seeds"]]
    for seed in seeds:
        for name, k, init_kind, label in CONDITIONS:
            json_path, npz_path = out / "runs" / f"{name}_seed{seed}.json", out / "runs" / f"{name}_seed{seed}.npz"
            if json_path.exists() and npz_path.exists():
                saved = json.loads(json_path.read_text())
                rows.append(saved["row"]); trajectories[(name, seed)] = saved["trajectory"]
                if name == "k1_standard": baseline_var[seed] = float(np.load(npz_path)["variances"].mean())
                log(f"[exp3] reuse {name}, seed {seed}")
                continue
            rng = np.random.default_rng(seed)
            if init_kind == "standard":
                init_means, init_vars = target.initial_mixture(k, rng)
                init_note = "repository prior-free conditional initialisation"
            elif init_kind == "spread":
                init_means, init_vars = _spread_init(target, seed, baseline_var[seed],
                                                     float(exp["spread_radius_scale"]))
                init_note = "opposite random directions; no teacher samples or labels"
            else:
                init_means, init_vars = oracle_means.copy(), np.full(2, baseline_var[seed])
                init_note = f"DINOv2 k-means representatives {oracle_indices}; oracle control"
            trace, callback = _trajectory_callback(init_means, init_vars,
                                                   int(exp["n_iterations"]),
                                                   int(exp["trace_every"]),
                                                   float(exp["collapse_threshold"]))
            log(f"[exp3] fit {label}, seed {seed}")
            fit = ibw_gpu.fit(target, n_components=k, method="ibw",
                              n_iterations=int(exp["n_iterations"]),
                              step_size=float(exp["step_size"]),
                              n_grad_samples=int(exp["n_grad_samples"]),
                              init_means=init_means, init_variances=init_vars,
                              seed=seed, device=dev, callback=callback)
            if name == "k1_standard": baseline_var[seed] = float(np.mean(fit["variances"]))
            quality = _evaluate(fit["means"], fit["variances"], teacher, target,
                                teacher_features, exp, seed + 1000 * (CONDITIONS.index((name, k, init_kind, label)) + 1))
            sep = _component_stats(fit["means"], fit["variances"], float(exp["collapse_threshold"]))
            init_sep = _component_stats(init_means, init_vars, float(exp["collapse_threshold"]))
            row = dict(condition=name, label=label, k=k, init=init_kind, seed=seed,
                       initial_normalized_separation_mean=init_sep["normalized_separation_mean"],
                       final_variance_mean=float(np.mean(fit["variances"])),
                       grad_norm_first=float(fit["grad_norms"][0]),
                       grad_norm_last=float(fit["grad_norms"][-1]),
                       runtime_s=float(fit["runtime_s"]), stopped_early=fit["stopped_early"],
                       **sep, **quality)
            rows.append(row); trajectories[(name, seed)] = trace
            np.savez_compressed(npz_path, means=fit["means"], variances=fit["variances"],
                                init_means=init_means, init_variances=init_vars,
                                variances_trace=fit["variances_trace"], grad_norms=fit["grad_norms"])
            Pv.dump_json(dict(row=row, trajectory=trace, init_note=init_note,
                              settings=fit["settings"]), json_path)
            log(f"[exp3]   MMD²={row['mmd2']:.4f}, SWD={row['sliced_wasserstein']:.4f}, "
                f"separation={row['normalized_separation_mean']}")

    _write_csv(rows, out / "numerical" / "run_metrics.csv")
    result = dict(config=cfg, target=target.describe(), oracle=dict(indices=oracle_indices,
                  cluster_sizes=oracle_sizes.tolist()), collapse_definition=(
                  "collapsed iff any pairwise center distance / (sigma_i + sigma_j) is below "
                  f"{exp['collapse_threshold']}; raw and normalized distances are both retained"),
                  rows=rows, trajectories={f"{k[0]}_seed{k[1]}": v for k, v in trajectories.items()})
    Pv.dump_json(result, out / "numerical" / "results.json")
    _figures(rows, trajectories, out)

    lines = ["# Experiment 3 — IBW initialisation diagnostic", "",
             "Target: continuous-time VE Bayes conditional for the canonical H1 0.60→0.45 state.",
             "Teacher-cluster initialisation is an oracle diagnostic, not a practical method.", "",
             "| condition | mean DINOv2 MMD² | mean SWD | mean normalized separation |", "|---|---:|---:|---:|"]
    for name, _, _, label in CONDITIONS:
        rr = [r for r in rows if r["condition"] == name]
        sep = [r["normalized_separation_mean"] for r in rr
               if r["normalized_separation_mean"] is not None]
        lines.append(f"| {label} | {np.mean([r['mmd2'] for r in rr]):.5f} | "
                     f"{np.mean([r['sliced_wasserstein'] for r in rr]):.5f} | "
                     f"{np.mean(sep) if sep else float('nan'):.3f} |")
    lines += ["", "See `numerical/results.json` for raw distances, collapse flags, and trajectories."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    provenance = Pv.snapshot(cfg, dev)
    provenance.update(status="finished", runtime_s=time.time() - started,
                      reference=str(reference), target=target.describe())
    Pv.dump_json(provenance, out / "provenance.json")
    log(f"[exp3] complete -> {out}")
    return rows


def replot(cfg, out_dir=None):
    """Regenerate aggregate figures from completed scalar results and trajectories."""
    out = Path(out_dir or ROOT / cfg["output"]["dir"])
    payload = json.loads((out / "numerical" / "results.json").read_text())
    trajectories = {}
    for key, value in payload["trajectories"].items():
        condition, seed = key.rsplit("_seed", 1)
        trajectories[(condition, int(seed))] = value
    _figures(payload["rows"], trajectories, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="exp3_ibw_initialization")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args(argv)
    cfg = C.load_config(args.config)
    return replot(cfg, out_dir=args.out) if args.plot_only else run(
        cfg, out_dir=args.out, device=args.device)


if __name__ == "__main__":
    main()
