"""GMMD / conditional_multimodality -- H1: is ``p_theta(x_s | x_t)`` multimodal for ONE fixed ``x_t``?

For one held-out CelebA-HQ image ``x_0``:

1. ``x_t = x_0 + sigma(t) z`` from the teacher's forward marginal, ONE draw, then frozen;
2. ``N`` independent stochastic reverse-SDE rollouts ``x_t -> x_s`` from that same tensor
   (per-rollout seeds), giving ``S(x_t) = {x_s^(1), ..., x_s^(N)}``;
3. control 1: the probability-flow ODE from the same ``x_t``, repeated -- must not vary;
   control 2: the same rollouts observed at a nearby time ``s_near`` -- a short transition;
4. every sample is also shown through the teacher's own denoiser ``E_theta[x_0 | x_s]``
   (Tweedie, one network call) because a raw ``x_s`` at sigma ~ 1 is unreadable;
5. features, PCA / UMAP views, and the multimodality DIAGNOSTICS of :mod:`gmmd.diagnostics`
   for the long transition, the short one and the deterministic control, side by side.

Everything is written under ``results/<output.dir>/``: ``config.json`` (resolved),
``provenance.json`` (git commit, packages, GPU, checkpoint digest, dataset, image id, seeds,
t / s as realised on the grid, sample counts, NFE, runtimes), ``samples.npz`` (x_0, x_t, every
x_s and its denoising, controls), ``features.npz``, ``diagnostics.json`` and the figures.

Run::

    CUDA_VISIBLE_DEVICES=<free gpu> python -m gmmd.experiments.conditional_multimodality \\
        conditional_multimodality [--set transition.n_samples=8] [--set features.extractor=pixel]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from gmmd import config as C
from gmmd import diagnostics as D
from gmmd import features as Fe
from gmmd import plotting as P
from gmmd import provenance as Pv
from gmmd.datasets.celebahq import CelebAHQ256, data_root

__all__ = ["run", "main"]


def _digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _batched(fn, x, batch_size):
    return np.concatenate([fn(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def _strip(d):
    """diagnostics dict -> json-sized (drop per-point arrays, keep the null draws)."""
    out = {}
    for k, v in d.items():
        if k in ("pca_scores",):
            continue
        if isinstance(v, dict):
            v = _strip(v)
            v.pop("labels", None)
        out[k] = v
    return out


def run(cfg, device=None, out_dir=None, log=print):
    import torch
    from gmmd.teachers.score_sde.teacher import ScoreSDETeacher

    T0 = time.time()
    timings = {}
    name = cfg["name"]
    out = Path(out_dir) if out_dir else C.RESULTS_DIR / cfg["output"]["dir"]
    out.mkdir(parents=True, exist_ok=True)
    root = data_root()
    dev = device or cfg["teacher"]["device"]
    prov = Pv.snapshot(cfg, dev)
    Pv.dump_json(cfg, out / "config.json")
    Pv.dump_json(dict(prov, status="started"), out / "provenance.json")

    # ---- data: one held-out image ------------------------------------------------------------
    dcfg = cfg["dataset"]
    ds = CelebAHQ256(root=C.resolve_path(dcfg["root"], root), source=dcfg["source"],
                     split=dcfg["split"], image_size=dcfg["image_size"])
    idx = int(dcfg["image_index"])
    x0 = ds[idx]
    image_id = ds.identifier(idx)
    log(f"[{name}] image {idx} = {image_id}  x0 {x0.shape} in [{x0.min():.3f}, {x0.max():.3f}]")

    # ---- teacher ------------------------------------------------------------------------------
    tcfg = cfg["teacher"]
    t1 = time.time()
    teacher = ScoreSDETeacher.from_config(tcfg["config"], C.resolve_path(tcfg["checkpoint"], root),
                                         device=dev, use_ema=bool(tcfg.get("use_ema", True)),
                                         allow_tf32=bool(tcfg.get("allow_tf32", False)))
    timings["teacher_load_s"] = time.time() - t1
    sampler = dict(predictor=tcfg["predictor"], corrector=tcfg["corrector"], snr=float(tcfg["snr"]),
                   n_steps_each=int(tcfg["n_steps_each"]))
    bs = int(tcfg["batch_size"])
    log(f"[{name}] teacher {tcfg['config']} step {teacher.info['step']} ema={teacher.info['use_ema']} "
        f"on {dev}; sampler {sampler}")

    # ---- x_t: one forward draw, then frozen -------------------------------------------------
    tr = cfg["transition"]
    i_t, i_s, tg, sg = teacher.transition_indices(tr["t"], tr["s"])
    i_near, s_near_g = teacher.grid_index(tr["s_near"])
    if not (i_t < i_near <= i_s):
        raise ValueError(f"s_near={tr['s_near']} must satisfy s <= s_near < t (grid indices "
                         f"{i_t} < {i_near} <= {i_s}); the short transition is the same rollouts "
                         "observed earlier")
    x_t, xt_info = teacher.forward_noise(x0, tg, int(tr["forward_noise_seed"]))
    x_t = x_t.detach().clone()                      # THE frozen tensor every rollout starts from
    x_t_np = x_t.cpu().numpy()
    x_t_digest = _digest(x_t_np)
    x_t_den = teacher.tweedie_denoise(x_t, tg)[0].cpu().numpy()
    sig_t, sig_s, sig_near = teacher.sigma(tg), teacher.sigma(sg), teacher.sigma(s_near_g)
    log(f"[{name}] t={tg:.5f} (sigma {sig_t:.3g}, grid {i_t}) -> s={sg:.5f} (sigma {sig_s:.3g}, "
        f"grid {i_s}): {i_s - i_t} steps; s_near={s_near_g:.5f} (sigma {sig_near:.3g}, grid {i_near}); "
        f"x_t sha256 {x_t_digest[:12]}")

    # ---- N stochastic rollouts from the SAME x_t ---------------------------------------------
    N = int(tr["n_samples"])
    seeds = [int(tr["rollout_seed_base"]) + i for i in range(N)]
    xs, xnear, inter, inter_steps = [], [], [], None
    calls, t1 = 0, time.time()
    nfe_per_sample = None
    for b in range(0, N, bs):
        res = teacher.sample_transition(x_t, tg, sg, seeds[b:b + bs], record_at=[i_near],
                                        record_every=int(tr.get("record_every", 0)), **sampler)
        k = int(np.where(res.intermediate_steps == i_near)[0][0])
        xs.append(res.x_s_np)
        xnear.append(res.intermediates[k])
        if int(tr.get("record_every", 0)):
            inter.append(res.intermediates)
            inter_steps = res.intermediate_steps
        calls += res.n_forward_calls
        nfe_per_sample = res.nfe_per_sample
        log(f"[{name}]   rollouts {b + 1}-{b + len(res.seeds)}/{N}: {res.n_steps} steps, "
            f"{res.nfe_per_sample} NFE each, {res.runtime_s:.1f}s")
        if not np.isfinite(res.x_s_np).all():
            raise FloatingPointError("non-finite x_s in a rollout batch")
    timings["rollouts_s"] = time.time() - t1
    x_s = np.concatenate(xs)
    x_near = np.concatenate(xnear)
    assert np.array_equal(x_t.cpu().numpy(), x_t_np), "x_t was modified during the rollouts"

    # ---- the teacher's denoiser on every sample (visualisation + feature input) --------------
    t1 = time.time()
    den = lambda X, t: _batched(lambda v: teacher.tweedie_denoise(torch.as_tensor(v), t).cpu().numpy(), X, bs)
    x_s_den = den(x_s, sg)
    x_near_den = den(x_near, s_near_g)
    timings["denoise_s"] = time.time() - t1

    # ---- control 1: the probability-flow ODE, repeated ----------------------------------------
    t1 = time.time()
    R = int(tr["deterministic_repeats"])
    dets = [teacher.ode_transition(x_t, tg, sg, predictor=sampler["predictor"], n_copies=1).x_s_np[0]
            for _ in range(R)]
    x_det = np.stack(dets)
    det_diff = float(max(np.abs(x_det[i] - x_det[j]).max() for i in range(R) for j in range(i + 1, R))) if R > 1 else 0.0
    x_det_den = den(x_det, sg)
    timings["deterministic_control_s"] = time.time() - t1
    log(f"[{name}] control 1: {R} ODE repeats, max |diff| = {det_diff:.3e}")

    # ---- features -----------------------------------------------------------------------------
    fcfg = cfg["features"]
    ext = fcfg["extractor"]
    t1 = time.time()
    raw = dict(long=x_s, near=x_near, deterministic=x_det)
    dnz = dict(long=x_s_den, near=x_near_den, deterministic=x_det_den)
    feats = {}
    for inp, groups in (("raw", raw), ("denoised", dnz)):
        for g, X in groups.items():
            feats[f"pixel/{inp}/{g}"] = Fe.extract("pixel", X, downsample=int(fcfg["pixel_downsample"]))
            if ext != "pixel":
                feats[f"{ext}/{inp}/{g}"] = Fe.extract(ext, X, device=dev, batch_size=bs)
    np.savez_compressed(out / "features.npz", **feats)
    timings["features_s"] = time.time() - t1
    rep = f"{ext}/{fcfg['input']}"

    # ---- diagnostics ----------------------------------------------------------------------------
    dg = cfg["diagnostics"]
    t1 = time.time()
    kw = dict(pca_dims=int(dg["pca_dims"]), max_k=int(dg["gmm_max_components"]),
              covariance_type=dg["gmm_covariance"], cv_folds=int(dg["cv_folds"]),
              n_boot=int(dg["n_boot"]), n_null=int(dg["n_null"]), seed=int(dg["seed"]))
    diags = {g: D.analyze(feats[f"{rep}/{g}"], **kw) for g in ("long", "near", "deterministic")}
    pixel_space = {g: D.spread_stats(X.reshape(len(X), -1)) for g, X in raw.items()}
    pixel_space_denoised = {g: D.spread_stats(X.reshape(len(X), -1)) for g, X in dnz.items()}
    ref = None
    if rep != "pixel/raw":
        ref = {g: D.analyze(feats[f"pixel/raw/{g}"], **kw) for g in ("long", "near")}
    timings["diagnostics_s"] = time.time() - t1
    lines = []
    for g, d in diags.items():
        lines += D.summarize(d, f"{rep} {g}")
    if ref:
        for g, d in ref.items():
            lines += D.summarize(d, f"pixel/raw {g}")
    for ln in lines:
        log(ln)

    # ---- 2-D views: one joint PCA and one joint UMAP over the three groups --------------------
    t1 = time.time()
    order = ["long", "near", "deterministic"]
    Z = np.concatenate([feats[f"{rep}/{g}"] for g in order]).astype(np.float64)
    sizes = [len(feats[f"{rep}/{g}"]) for g in order]
    cuts = np.cumsum([0] + sizes)
    from sklearn.decomposition import PCA
    E = PCA(n_components=2, svd_solver="full", random_state=int(dg["seed"])).fit_transform(Z)
    emb = {"pca": {g: E[cuts[i]:cuts[i + 1]] for i, g in enumerate(order)}}
    ucfg = cfg.get("umap", {})
    umap_info = None
    if ucfg.get("enabled", True):
        import umap
        nn = int(min(ucfg.get("n_neighbors", 15), len(Z) - 1))
        ukw = dict(n_components=2, n_neighbors=nn, min_dist=float(ucfg.get("min_dist", 0.1)),
                   metric="euclidean", random_state=int(ucfg.get("seed", 0)))
        U = np.asarray(umap.UMAP(**ukw).fit_transform(Z), float)
        emb["umap"] = {g: U[cuts[i]:cuts[i + 1]] for i, g in enumerate(order)}
        umap_info = dict(ukw, umap_version=umap.__version__)
    timings["embeddings_s"] = time.time() - t1

    # ---- figures ---------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = dict(t=tg, s=sg, s_near=s_near_g, sigma_t=sig_t, sigma_s=sig_s, sigma_near=sig_near,
                image_id=image_id, forward_noise_seed=int(tr["forward_noise_seed"]),
                seed_first=seeds[0], seed_last=seeds[-1], n_steps=i_s - i_t,
                nfe_per_sample=nfe_per_sample, det_max_abs_diff=det_diff,
                sampler=f"{sampler['predictor']}" + (f" + {sampler['corrector']} corrector" if sampler["corrector"] != "none" else " (reverse SDE, no corrector)"))
    figs = {
        "sample_grid": P.sample_grid_figure(x0, x_t_np, x_t_den, x_s, x_s_den, meta),
        "controls": P.controls_figure(x_near, x_near_den, x_det, x_det_den, meta),
        "embedding_pca": P.embedding_figure(emb["pca"], "PC", f"joint PCA of {rep} features"),
        "gmm_selection": P.selection_figure(diags),
        "lrt_null": P.lrt_figure(diags),
    }
    if "umap" in emb:
        figs["embedding_umap"] = P.embedding_figure(emb["umap"], "UMAP", f"joint UMAP of {rep} features (a view, not a test)")
    for stem, fig in figs.items():
        P.savefig(fig, out / stem)
        plt.close(fig)

    # ---- persist -----------------------------------------------------------------------------------
    if cfg["output"].get("save_samples", True):
        arrays = dict(x0=x0, x_t=x_t_np, x_t_denoised=x_t_den, x_s=x_s, x_s_denoised=x_s_den,
                      x_near=x_near, x_near_denoised=x_near_den, x_det=x_det, x_det_denoised=x_det_den,
                      rollout_seeds=np.asarray(seeds), t=tg, s=sg, s_near=s_near_g,
                      grid_indices=np.asarray([i_t, i_near, i_s]))
        if inter:
            arrays["intermediates"] = np.concatenate(inter, axis=1)
            arrays["intermediate_steps"] = inter_steps
        np.savez_compressed(out / "samples.npz", **arrays)
    np.savez_compressed(out / "embeddings.npz", **{f"{m}/{g}": v for m, d in emb.items() for g, v in d.items()})
    diag_out = dict(representation=rep, feature_dims={k: int(v.shape[1]) for k, v in feats.items()},
                    groups={g: _strip(d) for g, d in diags.items()},
                    pixel_space_raw=pixel_space, pixel_space_denoised=pixel_space_denoised,
                    reference_pixel_raw={g: _strip(d) for g, d in ref.items()} if ref else None,
                    deterministic_control=dict(repeats=R, max_abs_diff=det_diff),
                    summary_lines=lines, umap=umap_info, settings=kw)
    Pv.dump_json(diag_out, out / "diagnostics.json")
    timings["total_s"] = time.time() - T0
    prov.update(status="finished", teacher=teacher.describe(), dataset=ds.describe(),
                image=dict(index=idx, identifier=image_id, x0_sha256=_digest(x0)),
                transition=dict(t=tg, s=sg, s_near=s_near_g, i_t=i_t, i_s=i_s, i_near=i_near,
                                n_steps=i_s - i_t, sigma_t=sig_t, sigma_s=sig_s, sigma_near=sig_near,
                                t_requested=tr["t"], s_requested=tr["s"], s_near_requested=tr["s_near"]),
                x_t=dict(sha256=x_t_digest, **xt_info), sampler=sampler, n_samples=N,
                rollout_seeds=seeds, batch_size=bs, nfe_per_sample=nfe_per_sample,
                n_network_forwards=int(calls + teacher.n_score_calls - calls),
                deterministic_control=dict(repeats=R, max_abs_diff=det_diff),
                features=dict(extractor=ext, input=fcfg["input"], representation=rep),
                timings=timings, outputs=sorted(p.name for p in out.iterdir()))
    Pv.dump_json(prov, out / "provenance.json")
    log(f"[{name}] done in {timings['total_s']:.0f}s -> {out}")
    return dict(out_dir=out, diagnostics=diag_out, provenance=prov)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gmmd.experiments.conditional_multimodality",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="config name under configs/ (e.g. conditional_multimodality)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value (dotted key, JSON value), repeatable")
    ap.add_argument("--device", default=None, help="torch device (default: the config's)")
    ap.add_argument("--out", default=None, help="output directory (default results/<output.dir>)")
    args = ap.parse_args(argv)
    cfg = C.load_config(args.config, C.parse_overrides(args.set))
    return run(cfg, device=args.device, out_dir=args.out)


if __name__ == "__main__":
    main()
