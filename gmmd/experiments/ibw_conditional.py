"""How many isotropic Gaussians does it take to reproduce ``p_θ(x_s | x_t)``?

Fits the IBW mixture of Petit-Talamon, Lambert & Korba to the conditional of a *finished* H1 run
at a sweep of component counts ``K``, and asks -- for each ``K`` -- whether samples from the
fitted ``q`` are distinguishable from the teacher's own conditional samples.

    python -m gmmd.experiments.ibw_conditional results/conditional_multimodality --target near

**What the target is.**  :class:`gmmd.conditional_target.VEConditionalTarget`, i.e. the
closed-form conditional score of the continuous-time reversal,
``s_θ(x_s, s) + (x_t - x_s)/(σ_t² - σ_s²)``.  That is *not* the discretised sampler's transition
kernel, which has no score.  Read that module's docstring before reading these numbers: the
whole design of this experiment is to measure the gap rather than assume it away, which is what
the two-sample test against the teacher's saved samples does.

**Why the objective is not reported.**  Only the conditional's score is available, never its
log-density, so ``KL(q | p)`` can be descended but not evaluated.  Model order is therefore
selected on samples: for each ``K``, an MMD permutation test of ``q``'s samples against the
teacher's.  A large p-value means "no detectable difference at n = 64", which at this sample
size is weak evidence and is reported as such.

**Protocol.**  The step size is tuned ONCE, at ``K = 1``, and then held fixed for every other
``K``, so the comparison across ``K`` is not contaminated by per-``K`` tuning.  Initialisation is
the prior-free conditional ``N(x_t, (σ_t² - σ_s²) I)``: at d = 196 608 a uniform draw would put
every component in a region the score network has never seen.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from gmmd import diagnostics as D
from gmmd import features as Fe
from gmmd import ibw, ibw_gpu
from gmmd import provenance as Pv
from gmmd.conditional_target import VEConditionalTarget
from gmmd.config import resolve_path
from gmmd.datasets.celebahq import data_root
from gmmd.plotting import savefig, tile

__all__ = ["run", "main"]

DEFAULT_STEP_SIZES = (0.01, 0.03, 0.1)
DEFAULT_COMPONENTS = (1, 2, 4, 8)


def _run_config(run_dir):
    """The resolved config the run was produced with, written next to its results."""
    import json

    return json.loads((Path(run_dir) / "config.json").read_text())


def _build_teacher(run_dir, device):
    from gmmd.teachers.score_sde.teacher import ScoreSDETeacher

    cfg = _run_config(run_dir)
    tc = cfg["teacher"]
    return ScoreSDETeacher.from_config(tc["config"], resolve_path(tc["checkpoint"], data_root()),
                                       device=device, use_ema=bool(tc.get("use_ema", True)),
                                       allow_tf32=bool(tc.get("allow_tf32", False))), cfg


def _fit_one(target, K, step_size, n_iterations, n_grad_samples, seed, log, tag="",
             init=None):
    """One IBW fit on the GPU backend.

    :mod:`gmmd.ibw_gpu` rather than the NumPy reference: at d = 196 608 the reference spends
    more time moving samples across PCIe and doing float64 linear algebra on the host than the
    score network spends on the GPU.  The two are pinned to each other in
    ``tests/test_ibw_gpu_parity.py``, so this inherits the reference's provenance.

    There is no objective to report (the conditional has no tractable log-density), so
    convergence is watched through the norm of the mean gradient: it falls to the Monte-Carlo
    noise floor at a stationary point, and a rising trace means the step size is too large.
    """
    rng = np.random.default_rng(seed)
    init_means, init_vars = target.initial_mixture(K, rng) if init is None else init
    res = ibw_gpu.fit(target, n_components=K, method="ibw", n_iterations=n_iterations,
                      step_size=step_size, n_grad_samples=n_grad_samples,
                      init_means=init_means, init_variances=init_vars, seed=seed,
                      device=target.device)
    g = res["grad_norms"]
    log(f"[ibw_conditional] {tag}K={K} gamma={step_size:g}: {res['runtime_s']:.0f}s, "
        f"|grad_m| {g[0]:.2f} -> {g[-1]:.2f}, "
        f"variances {np.round(res['variances'], 3).tolist()}, "
        f"mean |m - x_t| = {np.abs(res['means'] - target.x_t[None]).mean():.4f}"
        + (f"  STOPPED: {res['stopped_early']}" if res["stopped_early"] else ""))
    return res


def teacher_mode_init(teacher_raw, teacher_denoised, K, variance, feature_extractor, device,
                      batch_size, seed, log):
    """Initial means placed ON the teacher's own conditional modes.

    The decisive diagnostic for a collapse.  If IBW merges components because every one of them
    started in the same basin, then starting them in DIFFERENT basins -- at K representative
    teacher samples, chosen by k-means in feature space -- should keep them apart.  If they merge
    anyway, the target itself is unimodal, and the mismatch against the teacher's samples is the
    discretisation gap rather than an optimisation failure.

    The initial variance is the one the free fit converged to, so the ONLY thing that differs
    from the prior-free run is where the components start.
    """
    from sklearn.cluster import KMeans

    F = Fe.extract(feature_extractor, teacher_denoised, device=device, batch_size=batch_size)
    if K == 1:
        idx = [int(np.argmin(np.linalg.norm(F - F.mean(0), axis=1)))]
    else:
        km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(F)
        # the real teacher sample closest to each cluster centre, so every mean is on the manifold
        idx = [int(np.argmin(np.linalg.norm(F - c, axis=1))) for c in km.cluster_centers_]
        log(f"[ibw_conditional] mode-init K={K}: cluster sizes "
            f"{np.bincount(km.labels_, minlength=K).tolist()}, representative samples {idx}")
    means = teacher_raw[idx].reshape(len(idx), -1).astype(np.float64)
    return means, np.full(len(idx), float(variance))


def _evaluate(res, target, teacher, teacher_raw, teacher_denoised, n_eval, seed, batch_size,
              feature_extractor, device, n_perm, log, tag=""):
    """Sample the fitted q and two-sample-test it against the teacher's conditional samples."""
    q = ibw.IsotropicMixture(res["means"], res["variances"])
    xq = q.sample(n_eval, np.random.default_rng(seed + 1))
    xq_img = xq.reshape(-1, *target.image_shape).astype(np.float32)
    if not np.isfinite(xq_img).all():
        raise FloatingPointError("the fitted mixture produced non-finite samples; the step size "
                                 "is too large or the fit diverged")

    import torch
    den = np.concatenate([
        teacher.tweedie_denoise(torch.as_tensor(xq_img[i:i + batch_size]), target.s).cpu().numpy()
        for i in range(0, len(xq_img), batch_size)])

    out = dict(n_eval=int(n_eval))
    # raw pixels: q and the teacher live in the same space at time s, directly comparable
    out["pixel"] = D.mmd_permutation_test(Fe.extract("pixel", xq_img),
                                          Fe.extract("pixel", teacher_raw), n_perm=n_perm, seed=seed)
    # semantic features: both sides pushed through the SAME denoiser first (see REPORT.md)
    out[feature_extractor] = D.mmd_permutation_test(
        Fe.extract(feature_extractor, den, device=device, batch_size=batch_size),
        Fe.extract(feature_extractor, teacher_denoised, device=device, batch_size=batch_size),
        n_perm=n_perm, seed=seed)
    # do the components stay apart, or collapse onto one another?
    m = res["means"]
    if len(m) > 1:
        from scipy.spatial.distance import pdist
        sep = pdist(m)
        scale = float(np.sqrt(res["variances"].mean()))
        out["separation"] = dict(min=float(sep.min()), mean=float(sep.mean()),
                                 max=float(sep.max()), sigma=scale,
                                 min_over_sigma=float(sep.min() / scale))
    out["variances"] = res["variances"].tolist()
    for rep in (feature_extractor, "pixel"):
        r = out[rep]
        log(f"[ibw_conditional] {tag}  {rep:16s} MMD^2 = {r['mmd2']:+.5f}  "
            f"(null q95 {r['null_q95']:+.5f})  p = {r['p_value']:.3f}")
    if "separation" in out:
        log(f"[ibw_conditional] {tag}  component separation / sigma: "
            f"min {out['separation']['min_over_sigma']:.2f}")
    return out, den


def run(run_dir, target_time="near", components=DEFAULT_COMPONENTS, step_sizes=DEFAULT_STEP_SIZES,
        n_iterations=300, n_grad_samples=4, n_eval=64, n_perm=500, seed=0, device=None,
        batch_size=16, feature_extractor="dinov2_vitb14", feature_device="cpu",
        init="prior_free", log=print):
    run_dir = Path(run_dir)
    z = np.load(run_dir / "samples.npz")
    device = device or _run_config(run_dir)["teacher"]["device"]
    teacher, _ = _build_teacher(run_dir, device)
    t = float(z["t"])
    s = float(z["s_near"]) if target_time == "near" else float(z["s"])
    teacher_raw = z["x_near"] if target_time == "near" else z["x_s"]
    teacher_den = z["x_near_denoised"] if target_time == "near" else z["x_s_denoised"]

    tgt = VEConditionalTarget(teacher, z["x_t"], t, s, batch_size=batch_size)
    log(f"[ibw_conditional] target: {tgt.describe()['definition']}")
    log(f"[ibw_conditional] t={tgt.t:.5f} (sigma {tgt.sigma_t:.3f}) -> s={tgt.s:.5f} "
        f"(sigma {tgt.sigma_s:.3f});  gap = sigma_t^2 - sigma_s^2 = {tgt.gap:.3f};  d = {tgt.dim}")
    log(f"[ibw_conditional] teacher reference: {len(teacher_raw)} samples from the {target_time} "
        f"transition of {run_dir.name}")

    # ---- step size: tuned once at K = 1, then frozen -------------------------------------
    log("[ibw_conditional] --- step-size selection at K = 1 (held fixed for every K) ---")
    scan = {}
    for g in step_sizes:
        r = _fit_one(tgt, 1, g, n_iterations, n_grad_samples, seed, log, tag="scan ")
        ev, _ = _evaluate(r, tgt, teacher, teacher_raw, teacher_den, n_eval, seed, batch_size,
                          feature_extractor, feature_device, n_perm, log, tag="scan ")
        scan[g] = dict(fit=_slim(r), eval=ev)
    best = min(scan, key=lambda g: scan[g]["eval"][feature_extractor]["mmd2"])
    log(f"[ibw_conditional] selected gamma = {best:g} (lowest MMD^2 in {feature_extractor} at K=1)")

    # ---- the sweep -------------------------------------------------------------------------
    log(f"[ibw_conditional] --- K sweep at gamma = {best:g}, init = {init} ---")
    fitted_variance = float(np.mean(scan[best]["fit"]["variances"]))
    results, denoised = {}, {}
    for K in components:
        init_kw = None if init == "prior_free" else teacher_mode_init(
            teacher_raw, teacher_den, K, fitted_variance, feature_extractor, feature_device,
            batch_size, seed, log)
        if init_kw is not None:
            from scipy.spatial.distance import pdist
            d0 = pdist(init_kw[0])
            log(f"[ibw_conditional] K={K} mode-init separation / sigma: "
                f"{(d0.min() / np.sqrt(fitted_variance)):.2f}" if K > 1 else
                f"[ibw_conditional] K={K} mode-init (single representative)")
        r = _fit_one(tgt, K, best, n_iterations, n_grad_samples, seed, log, init=init_kw)
        ev, den = _evaluate(r, tgt, teacher, teacher_raw, teacher_den, n_eval, seed, batch_size,
                            feature_extractor, feature_device, n_perm, log, tag=f"K={K} ")
        results[K] = dict(fit=_slim(r), eval=ev)
        denoised[K] = den

    _figures(run_dir, results, denoised, teacher_den, components, feature_extractor,
             target_time if init == "prior_free" else f"{target_time}_{init}", log)
    out = dict(run=str(run_dir), target=tgt.describe(), target_time=target_time,
               teacher_n=int(len(teacher_raw)), step_size_scan={str(k): v for k, v in scan.items()},
               selected_step_size=best, components=list(components),
               results={str(k): v for k, v in results.items()}, init=init,
               settings=dict(n_iterations=n_iterations, n_grad_samples=n_grad_samples,
                             n_eval=n_eval, n_perm=n_perm, seed=seed,
                             feature_extractor=feature_extractor),
               n_score_calls=int(tgt.n_score_calls), provenance=Pv.snapshot(None, device))
    suffix = target_time if init == "prior_free" else f"{target_time}_{init}"
    Pv.dump_json(out, run_dir / f"ibw_conditional_{suffix}.json")
    log(f"[ibw_conditional] -> {run_dir / f'ibw_conditional_{suffix}.json'}")
    return out


def _slim(res):
    g = np.asarray(res["grad_norms"], dtype=float)
    return dict(means_norm=float(np.linalg.norm(res["means"])), variances=res["variances"].tolist(),
                stopped_early=res["stopped_early"], runtime_s=res["runtime_s"],
                n_iterations_run=int(res["n_iterations_run"]), settings=res["settings"],
                grad_norm_first=float(g[0]), grad_norm_last=float(g[-1]),
                grad_norm_trace=g[::max(1, len(g) // 40)].tolist(),
                variances_trace=np.asarray(res["variances_trace"])[::max(1, len(g) // 40)].tolist())


def _figures(run_dir, results, denoised, teacher_den, components, rep, target_time, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(16, len(teacher_den))
    fig, axes = plt.subplots(len(components) + 1, 1, figsize=(16, 1.9 * (len(components) + 1) + 1),
                             constrained_layout=True)
    axes[0].imshow(tile(teacher_den[:n_show], ncols=n_show), interpolation="nearest")
    axes[0].set_title(f"teacher: {target_time} transition, {n_show} of the saved conditional samples",
                      fontsize=9)
    for ax, K in zip(axes[1:], components):
        p = results[K]["eval"][rep]["p_value"]
        ax.imshow(tile(denoised[K][:n_show], ncols=n_show), interpolation="nearest")
        ax.set_title(f"IBW fit, K = {K}   (MMD test vs teacher in {rep}: p = {p:.3f})", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    savefig(fig, run_dir / f"ibw_samples_{target_time}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    for r, c, mk in ((rep, "#2a78d6", "o"), ("pixel", "#eb6834", "s")):
        ax.plot(components, [results[K]["eval"][r]["mmd2"] for K in components], "-", marker=mk,
                color=c, ms=6, lw=1.6, label=r)
        ax.plot(components, [results[K]["eval"][r]["null_q95"] for K in components], ":", color=c,
                lw=1.2, label=f"{r}: null 95th pct")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("mixture components K"); ax.set_ylabel(r"unbiased MMD$^2$ vs teacher samples")
    ax.set_xscale("log", base=2); ax.set_xticks(components)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, color="0.92", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"{target_time} transition: below the dotted line = indistinguishable", fontsize=9)
    savefig(fig, run_dir / f"ibw_mmd_vs_K_{target_time}")
    plt.close(fig)
    log(f"[ibw_conditional] figures -> {run_dir}/ibw_*_{target_time}.png")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gmmd.experiments.ibw_conditional",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--target", default="near", choices=["near", "long"],
                    help="which transition of that run to fit (default: the short one)")
    ap.add_argument("--components", default=",".join(map(str, DEFAULT_COMPONENTS)))
    ap.add_argument("--step-sizes", default=",".join(map(str, DEFAULT_STEP_SIZES)))
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--grad-samples", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--feature-device", default="cpu")
    ap.add_argument("--init", default="prior_free", choices=["prior_free", "teacher_modes"],
                    help="where the components start: the prior-free conditional (default), or "
                         "ON the teacher's own modes (the collapse diagnostic)")
    a = ap.parse_args(argv)
    return run(a.run_dir, target_time=a.target,
               components=tuple(int(v) for v in a.components.split(",")),
               step_sizes=tuple(float(v) for v in a.step_sizes.split(",")),
               n_iterations=a.iterations, n_grad_samples=a.grad_samples, n_eval=a.n_eval,
               n_perm=a.n_perm, seed=a.seed, device=a.device, batch_size=a.batch_size,
               feature_device=a.feature_device, init=a.init)


if __name__ == "__main__":
    main()
