"""Experiment 2: controlled H1 test on an exact two-dimensional Gaussian mixture.

For each mode separation, a full-covariance sample fit (strong direct reference) and the
repository's uniform-isotropic IBW family are evaluated independently.  Reporting them as two
fit families separates representational benefit from IBW optimisation behaviour.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from gmmd import config as C
from gmmd import diagnostics as D
from gmmd import ibw
from gmmd import provenance as Pv
from gmmd.vi import isotropic_mixture_target
from gmmd.plotting import savefig

ROOT = Path(__file__).resolve().parents[2]


def _target_parameters(separation):
    d = float(separation)
    return np.asarray([[-d / 2.0, 0.0], [d / 2.0, 0.0]]), np.ones(2), np.full(2, 0.5)


def _sample(means, covariances, weights, n, rng):
    rng = np.random.default_rng(rng)
    labels = rng.choice(len(weights), size=int(n), p=np.asarray(weights) / np.sum(weights))
    out = np.empty((int(n), means.shape[1]))
    for k in range(len(weights)):
        idx = np.where(labels == k)[0]
        if len(idx):
            out[idx] = rng.multivariate_normal(means[k], covariances[k], size=len(idx))
    return out


def _log_prob(x, means, covariances, weights):
    x = np.asarray(x, float)
    parts = []
    for mean, cov, weight in zip(means, covariances, weights):
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("covariance must be positive definite")
        delta = x - mean
        quad = np.einsum("ni,ij,nj->n", delta, np.linalg.inv(cov), delta)
        parts.append(np.log(weight) - 0.5 * (x.shape[1] * np.log(2 * np.pi) + logdet + quad))
    return logsumexp(np.stack(parts, axis=1), axis=1)


def _direct_fits(train, seed):
    from sklearn.mixture import GaussianMixture

    mean = train.mean(0)
    cov = np.cov(train, rowvar=False, bias=True) + 1e-6 * np.eye(train.shape[1])
    q1 = dict(means=mean[None], covariances=cov[None], weights=np.ones(1))
    gm = GaussianMixture(n_components=2, covariance_type="full", n_init=10,
                         reg_covar=1e-6, random_state=seed, max_iter=1000).fit(train)
    q2 = dict(means=gm.means_, covariances=gm.covariances_, weights=gm.weights_)
    return q1, q2


def _ibw_fits(train, target, cfg, seed):
    from sklearn.cluster import KMeans

    out = []
    for k in (1, 2):
        if k == 1:
            labels = np.zeros(len(train), dtype=int)
            means = train.mean(0, keepdims=True)
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(train)
            labels, means = km.labels_, km.cluster_centers_
        variances = np.asarray([
            np.mean(np.sum((train[labels == j] - means[j]) ** 2, axis=1)) / train.shape[1]
            for j in range(k)
        ])
        variances = np.maximum(variances, 1e-4)
        result = ibw.fit(target, n_components=k, method="ibw",
                         n_iterations=int(cfg["n_iterations"]),
                         step_size=float(cfg["step_size"]),
                         n_grad_samples=int(cfg["n_grad_samples"]),
                         init_means=means, init_variances=variances, seed=seed,
                         kl_every=max(1, int(cfg["n_iterations"]) // 20),
                         n_kl_samples=int(cfg["n_kl_samples"]), store_trace=True)
        covs = result["variances"][:, None, None] * np.eye(train.shape[1])[None]
        model = dict(means=result["means"], covariances=covs, weights=result["weights"])
        out.append((model, result))
    return out


def _metrics(model, heldout, n_eval, n_mmd, n_projections, seed):
    rng = np.random.default_rng(seed)
    qsample = _sample(model["means"], model["covariances"], model["weights"], n_eval, rng)
    nll = float(-_log_prob(heldout, **model).mean())
    swd = D.sliced_wasserstein(qsample, heldout, n_projections=n_projections, seed=seed)
    m = min(int(n_mmd), len(qsample), len(heldout))
    qsub = qsample[rng.choice(len(qsample), m, replace=False)]
    psub = heldout[rng.choice(len(heldout), m, replace=False)]
    bandwidth = D.median_bandwidth(qsub, psub, seed=seed)
    mmd2 = D.mmd2_unbiased(qsub, psub, bandwidth)
    mode_mass_error = float(abs(np.mean(qsample[:, 0] >= 0.0) - 0.5))
    return dict(heldout_nll=nll, sliced_wasserstein=swd["mean"],
                sliced_wasserstein_projection_std=swd["std"], mmd2=mmd2,
                mmd_bandwidth=bandwidth, mode_mass_error=mode_mass_error)


def _write_csv(rows, path):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _mean_sem(rows, family, model, metric, separations):
    means, sems = [], []
    for d in separations:
        x = np.asarray([r[metric] for r in rows if r["fit_family"] == family
                        and r["model"] == model and np.isclose(r["separation"], d)])
        means.append(x.mean())
        sems.append(x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)
    return np.asarray(means), np.asarray(sems)


def _make_figures(rows, advantages, params, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Derive the plotted grid from completed rows so plot-only also works on deliberate smoke
    # subsets and on a partially completed diagnostic run.
    separations = sorted({float(r["separation"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    for family, label, color in (("direct_full_covariance", "sample direct fit", "#2a78d6"),
                                 ("ibw_isotropic", "IBW (isotropic)", "#eb6834")):
        means, sems = [], []
        for d in separations:
            x = np.asarray([r["sliced_wasserstein_advantage"] for r in advantages
                            if r["fit_family"] == family and np.isclose(r["separation"], d)])
            means.append(x.mean())
            sems.append(x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)
        ax.errorbar(separations, means, yerr=sems, marker="o", capsize=3, lw=1.8,
                    color=color, label=label)
    ax.axhline(0, color="0.5", ls="--", lw=1)
    ax.set_xlabel("target mode separation")
    ax.set_ylabel("GMM advantage in sliced Wasserstein\nSWD(K=1) − SWD(K=2); higher is better")
    ax.set_title("Mixture advantage as target multimodality increases")
    ax.grid(True, color="0.92", lw=0.7); ax.legend(frameon=False)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    savefig(fig, out / "figures" / "figure_2a_gmm_advantage")
    plt.close(fig)

    chosen = [separations[0], separations[len(separations) // 2], separations[-1]]
    first_seed = min(int(r["seed"]) for r in rows)
    grid = np.linspace(-6, 6, 180)
    xx, yy = np.meshgrid(grid, grid); points = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 9.0), sharex=True, sharey=True,
                             constrained_layout=True)
    for col, d in enumerate(chosen):
        target_means, target_vars, target_weights = _target_parameters(d)
        target = dict(means=target_means,
                      covariances=target_vars[:, None, None] * np.eye(2)[None],
                      weights=target_weights)
        models = [target, params[(d, first_seed, "direct_full_covariance", "K=1")],
                  params[(d, first_seed, "direct_full_covariance", "K=2")]]
        titles = ["ground-truth target", "fitted single Gaussian", "fitted two-component GMM"]
        samples = _sample(**target, n=350, rng=9000 + col)
        for row, (ax, model, title) in enumerate(zip(axes[:, col], models, titles)):
            z = np.exp(_log_prob(points, **model)).reshape(xx.shape)
            ax.contour(xx, yy, z, levels=7, cmap="Blues")
            ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.16, color="0.25")
            ax.scatter(model["means"][:, 0], model["means"][:, 1], marker="x", s=60,
                       color="#d62728", linewidth=2)
            if row == 0: ax.set_title(f"separation {d:g}\n{title}")
            else: ax.set_title(title)
            if col == 0: ax.set_ylabel("x₂")
            if row == 2: ax.set_xlabel("x₁")
            ax.set_aspect("equal")
    savefig(fig, out / "figures" / "figure_2b_distribution_snapshots")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    styles = {("direct_full_covariance", "K=1"): ("#2a78d6", "-", "direct K=1"),
              ("direct_full_covariance", "K=2"): ("#1baf7a", "-", "direct K=2"),
              ("ibw_isotropic", "K=1"): ("#9467bd", "--", "IBW K=1"),
              ("ibw_isotropic", "K=2"): ("#eb6834", "--", "IBW K=2")}
    for (family, model), (color, ls, label) in styles.items():
        mean, sem = _mean_sem(rows, family, model, "sliced_wasserstein", separations)
        ax.errorbar(separations, mean, yerr=sem, color=color, ls=ls, marker="o", capsize=2,
                    label=label)
    ax.set_xlabel("target mode separation"); ax.set_ylabel("sliced Wasserstein error (lower is better)")
    ax.set_title("Absolute approximation error")
    ax.grid(True, color="0.92", lw=0.7); ax.legend(frameon=False, ncol=2, fontsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    savefig(fig, out / "figures" / "figure_2c_error_vs_separation")
    plt.close(fig)


def run(cfg, out_dir=None, log=print):
    started = time.time()
    out = Path(out_dir or ROOT / cfg["output"]["dir"])
    for sub in ("numerical", "figures", "configs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    Pv.dump_json(cfg, out / "configs" / "resolved_config.json")
    exp, ocfg = cfg["experiment"], cfg["ibw"]
    rows, advantages, parameters, params_for_plot, traces = [], [], [], {}, {}
    for d in [float(v) for v in exp["separations"]]:
        target_means, target_vars, target_weights = _target_parameters(d)
        target = isotropic_mixture_target(target_means, target_vars, target_weights,
                                          name=f"symmetric_gmm_d={d:g}")
        target_covs = target_vars[:, None, None] * np.eye(2)[None]
        for seed in [int(v) for v in exp["seeds"]]:
            train = _sample(target_means, target_covs, target_weights, int(exp["n_train"]), seed)
            heldout = _sample(target_means, target_covs, target_weights, int(exp["n_test"]),
                              seed + 100_000)
            direct = _direct_fits(train, seed)
            ibw_models = _ibw_fits(train, target, ocfg, seed)
            for family, fitted in (("direct_full_covariance", [(direct[0], None), (direct[1], None)]),
                                   ("ibw_isotropic", ibw_models)):
                family_metrics = {}
                for k, (model, fit) in zip((1, 2), fitted):
                    metrics = _metrics(model, heldout, int(exp["n_eval"]), int(exp["n_mmd"]),
                                       int(exp["n_projections"]), seed + 1000 * k)
                    row = dict(separation=d, seed=seed, fit_family=family, model=f"K={k}", **metrics)
                    rows.append(row); family_metrics[k] = metrics
                    key = (d, seed, family, f"K={k}")
                    params_for_plot[key] = model
                    parameters.append(dict(separation=d, seed=seed, fit_family=family, model=f"K={k}",
                                           means=model["means"].tolist(),
                                           covariances=model["covariances"].tolist(),
                                           weights=model["weights"].tolist(),
                                           stopped_early=fit["stopped_early"] if fit else None,
                                           final_reverse_kl=float(fit["kls"][-1]) if fit is not None else None))
                    if fit is not None:
                        trace_key = f"d{d:g}_seed{seed}_K{k}".replace(".", "p")
                        traces[f"{trace_key}_means"] = fit["means_trace"]
                        traces[f"{trace_key}_variances"] = fit["variances_trace"]
                        traces[f"{trace_key}_kl"] = fit["kls"]
                        traces[f"{trace_key}_kl_iterations"] = fit["kl_iters"]
                advantages.append(dict(
                    separation=d, seed=seed, fit_family=family,
                    heldout_nll_advantage=family_metrics[1]["heldout_nll"] - family_metrics[2]["heldout_nll"],
                    sliced_wasserstein_advantage=family_metrics[1]["sliced_wasserstein"] - family_metrics[2]["sliced_wasserstein"],
                    mmd2_advantage=family_metrics[1]["mmd2"] - family_metrics[2]["mmd2"],
                    mode_mass_advantage=family_metrics[1]["mode_mass_error"] - family_metrics[2]["mode_mass_error"]))
            log(f"[exp2] separation={d:g}, seed={seed} complete")

    _write_csv(rows, out / "numerical" / "model_metrics.csv")
    _write_csv(advantages, out / "numerical" / "gmm_advantage.csv")
    Pv.dump_json(dict(config=cfg, metrics=rows, advantages=advantages, parameters=parameters),
                 out / "numerical" / "results.json")
    np.savez_compressed(out / "numerical" / "ibw_trajectories.npz", **traces)
    _make_figures(rows, advantages, params_for_plot, cfg, out)

    lines = ["# Experiment 2 — controlled H1 test", "",
             "Target: equal-weight 2D Gaussian mixture with identity component covariance.",
             "`direct_full_covariance` is the strong sample-fit reference; `ibw_isotropic` is the existing IBW family.", "",
             "| separation | fit family | mean SWD advantage (K=1 − K=2) |", "|---:|---|---:|"]
    for d in [float(v) for v in exp["separations"]]:
        for family in ("direct_full_covariance", "ibw_isotropic"):
            x = [r["sliced_wasserstein_advantage"] for r in advantages
                 if r["fit_family"] == family and np.isclose(r["separation"], d)]
            lines.append(f"| {d:g} | {family} | {np.mean(x):.5f} |")
    lines += ["", "Positive advantage means the two-component approximation is better."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    provenance = Pv.snapshot(cfg, "cpu")
    provenance.update(status="finished", runtime_s=time.time() - started)
    Pv.dump_json(provenance, out / "provenance.json")
    log(f"[exp2] complete -> {out}")
    return rows


def replot(cfg, out_dir=None):
    """Regenerate all aggregate figures from the completed JSON only."""
    out = Path(out_dir or ROOT / cfg["output"]["dir"])
    payload = json.loads((out / "numerical" / "results.json").read_text())
    params = {}
    for p in payload["parameters"]:
        key = (float(p["separation"]), int(p["seed"]), p["fit_family"], p["model"])
        params[key] = dict(means=np.asarray(p["means"]), covariances=np.asarray(p["covariances"]),
                           weights=np.asarray(p["weights"]))
    _make_figures(payload["metrics"], payload["advantages"], params, cfg, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="exp2_toy_h1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args(argv)
    cfg = C.load_config(args.config)
    return replot(cfg, out_dir=args.out) if args.plot_only else run(cfg, out_dir=args.out)


if __name__ == "__main__":
    main()
