"""Experiment 1: does H1 evidence strengthen with reverse-transition length?

Three frozen ``x_t`` states are followed along the same seeded reverse-SDE trajectories and
snapshotted at prespecified endpoints.  Every condition is analysed in DINOv2 space with the
existing H1 PCA/GMM/bootstrap pipeline.  Paired trajectories remove avoidable Monte-Carlo noise:
within an image, the only difference among endpoints is how long the same rollouts continued.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from gmmd import config as C
from gmmd import diagnostics as D
from gmmd import features as Fe
from gmmd import provenance as Pv
from gmmd.datasets.celebahq import CelebAHQ256, data_root
from gmmd.plotting import savefig

ROOT = Path(__file__).resolve().parents[2]


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _key(image_index, s):
    return f"image_{int(image_index):04d}_s_{float(s):.2f}".replace(".", "p")


def _strip(d):
    out = {}
    for k, v in d.items():
        if k == "pca_scores":
            continue
        if isinstance(v, dict):
            v = _strip(v)
            v.pop("labels", None)
        out[k] = v
    return out


def _denoise(teacher, images, s, batch_size):
    import torch

    return np.concatenate([
        teacher.tweedie_denoise(torch.as_tensor(images[i:i + batch_size]), s).cpu().numpy()
        for i in range(0, len(images), batch_size)
    ]).astype(np.float32)


def _row(record, diag):
    rows = diag["gmm"]["rows"]
    by_k = {int(r["K"]): r for r in rows}
    bic1, bic2 = by_k[1]["bic"], by_k[2]["bic"]
    st = diag["stability"]
    return dict(
        image_index=record["image_index"], image_id=record["image_id"],
        x_t_sha256=record["x_t_sha256"], forward_noise_seed=record["forward_noise_seed"],
        rollout_seed_first=record["seeds"][0],
        rollout_seed_last=record["seeds"][-1], n_samples=len(record["seeds"]),
        t=record["t"], s=record["s"], delta_t=record["t"] - record["s"],
        sigma_t=record["sigma_t"], sigma_s=record["sigma_s"], n_steps=record["n_steps"],
        preferred_k_bic=diag["gmm"]["best_k_bic"],
        preferred_k_cv=diag["gmm"]["best_k_cv"],
        delta_bic_2_minus_1=bic2 - bic1,
        cv_loglik_k1=by_k[1]["cv_loglik_mean"], cv_loglik_k2=by_k[2]["cv_loglik_mean"],
        cv_loglik_advantage_k2=by_k[2]["cv_loglik_mean"] - by_k[1]["cv_loglik_mean"],
        bootstrap_ari=st.get("ari_mean"), ari_null_p=st.get("p_ari"),
        silhouette=st.get("silhouette"), silhouette_null_p=st.get("p_silhouette"),
        lrt_p=diag["lrt"]["p_value"], pca_variance_explained=diag["pca"]["total_explained"],
    )


def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _figures(rows, condition_data, endpoints, image_indices, out, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    colors = plt.cm.tab10(np.linspace(0, 0.7, len(image_indices)))
    fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    for color, idx in zip(colors, image_indices):
        rr = sorted((r for r in rows if r["image_index"] == idx), key=lambda r: r["delta_t"])
        ax.plot([r["delta_t"] for r in rr], [r["delta_bic_2_minus_1"] for r in rr],
                "-o", color=color, alpha=0.75, label=f"fixed $x_t$: image {idx}")
    xs = sorted({r["delta_t"] for r in rows})
    means, sems = [], []
    for x in xs:
        y = np.asarray([r["delta_bic_2_minus_1"] for r in rows if np.isclose(r["delta_t"], x)])
        means.append(y.mean())
        sems.append(y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0.0)
    ax.errorbar(xs, means, yerr=sems, color="black", lw=2.4, marker="D", ms=6,
                capsize=3, label="mean ± SEM")
    ax.axhline(0, color="0.45", lw=1, ls="--")
    ax.set_xlabel(r"transition length  $\Delta t=t-s$")
    ax.set_ylabel(r"$\mathrm{BIC}_{K=2}-\mathrm{BIC}_{K=1}$  (lower favors two modes)")
    ax.set_title("Conditional multimodality versus reverse-transition length")
    ax.grid(True, color="0.92", lw=0.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    savefig(fig, out / "figures" / "figure_1a_multimodality_vs_length")
    plt.close(fig)

    matrix = np.asarray([[next(r["preferred_k_bic"] for r in rows
                               if r["image_index"] == idx and np.isclose(r["s"], s))
                          for s in endpoints] for idx in image_indices])
    fig, ax = plt.subplots(figsize=(6.4, 2.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap="viridis", vmin=1, vmax=max(4, matrix.max()), aspect="auto")
    ax.set_xticks(range(len(endpoints)), [f"s={s:.2f}\nΔt={0.60-s:.2f}" for s in endpoints])
    ax.set_yticks(range(len(image_indices)), [f"image {i}" for i in image_indices])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white",
                    fontweight="bold")
    ax.set_title("Preferred number of DINOv2-space components (BIC)")
    fig.colorbar(im, ax=ax, label="preferred K", ticks=range(1, max(4, matrix.max()) + 1))
    savefig(fig, out / "figures" / "figure_1b_preferred_k_heatmap")
    plt.close(fig)

    idx = image_indices[0]
    short, long = endpoints[0], endpoints[-1]
    a, b = condition_data[_key(idx, short)], condition_data[_key(idx, long)]
    pooled = np.concatenate([a["features"], b["features"]])
    E = PCA(n_components=2, svd_solver="full", random_state=seed).fit_transform(pooled)
    ea, eb = E[:len(a["features"])], E[len(a["features"]):]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), constrained_layout=True)
    for ax, e, dat, s, label in ((axes[0], ea, a, short, "short"),
                                 (axes[1], eb, b, long, "long")):
        lab = dat["labels"]
        for c in np.unique(lab):
            q = lab == c
            ax.scatter(e[q, 0], e[q, 1], s=28, alpha=0.85, edgecolor="white", lw=0.5,
                       label=f"cluster {int(c)} (n={q.sum()})")
        ax.set_title(f"{label} transition: 0.60 → {s:.2f}")
        ax.set_xlabel("joint DINOv2 PC 1"); ax.set_ylabel("joint DINOv2 PC 2")
        ax.grid(True, color="0.92", lw=0.6); ax.legend(frameon=False, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"Representative paired trajectories from fixed $x_t$ (image {idx})")
    savefig(fig, out / "figures" / "figure_1c_representative_embeddings")
    plt.close(fig)


def run(cfg, out_dir=None, device=None, log=print):
    import torch
    from gmmd.teachers.score_sde.teacher import ScoreSDETeacher

    started = time.time()
    out = Path(out_dir or ROOT / cfg["output"]["dir"])
    for sub in ("configs", "numerical", "samples", "embeddings", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    Pv.dump_json(cfg, out / "configs" / "resolved_config.json")
    Pv.dump_json(dict(Pv.snapshot(cfg, device), status="started"), out / "provenance.json")

    dcfg, tcfg, ecfg = cfg["dataset"], cfg["teacher"], cfg["experiment"]
    root = data_root()
    ds = CelebAHQ256(root=C.resolve_path(dcfg["root"], root), source=dcfg["source"],
                     split=dcfg["split"], image_size=int(dcfg["image_size"]))
    dev = device or tcfg["device"]
    teacher = ScoreSDETeacher.from_config(tcfg["config"], C.resolve_path(tcfg["checkpoint"], root),
                                         device=dev, use_ema=bool(tcfg.get("use_ema", True)),
                                         allow_tf32=bool(tcfg.get("allow_tf32", False)))
    sampler = dict(predictor=tcfg["predictor"], corrector=tcfg["corrector"],
                   snr=float(tcfg["snr"]), n_steps_each=int(tcfg["n_steps_each"]))
    image_indices = [int(v) for v in ecfg["image_indices"]]
    forward_seeds = [int(v) for v in ecfg["forward_noise_seeds"]]
    rollout_bases = [int(v) for v in ecfg["rollout_seed_bases"]]
    endpoints = [float(v) for v in ecfg["endpoints"]]
    if not (len(image_indices) == len(forward_seeds) == len(rollout_bases)):
        raise ValueError("image_indices, forward_noise_seeds and rollout_seed_bases must align")
    if endpoints != sorted(endpoints, reverse=True):
        raise ValueError("endpoints must be listed from shortest to longest transition")
    n, bs, t = int(ecfg["n_samples"]), int(tcfg["batch_size"]), float(ecfg["t"])

    condition_data = {}
    for image_index, fseed, rbase in zip(image_indices, forward_seeds, rollout_bases):
        x0, image_id = ds[image_index], ds.identifier(image_index)
        x_t, xt_info = teacher.forward_noise(x0, t, fseed)
        endpoint_info = [teacher.transition_indices(t, s) for s in endpoints]
        i_t, _, tg, _ = endpoint_info[0]
        index_to_s = {i_s: sg for _, i_s, _, sg in endpoint_info}
        seeds = list(range(rbase, rbase + n))
        gathered = {i_s: [] for i_s in index_to_s}
        log(f"[exp1] image {image_index}, forward seed {fseed}, rollout seeds {seeds[0]}..{seeds[-1]}")
        for start in range(0, n, bs):
            result = teacher.sample_transition(x_t, tg, endpoints[-1], seeds[start:start + bs],
                                               record_at=sorted(index_to_s), **sampler)
            for i_s in gathered:
                pos = int(np.where(result.intermediate_steps == i_s)[0][0])
                gathered[i_s].append(result.intermediates[pos])
            log(f"[exp1]   rollouts {start + 1}-{start + len(result.seeds)}/{n}")
        for _, i_s, _, sg in endpoint_info:
            raw = np.concatenate(gathered[i_s]).astype(np.float32)
            den = _denoise(teacher, raw, sg, bs)
            key = _key(image_index, sg)
            record = dict(key=key, image_index=image_index, image_id=image_id,
                          forward_noise_seed=fseed, seeds=seeds, t=tg, s=sg,
                          sigma_t=teacher.sigma(tg), sigma_s=teacher.sigma(sg),
                          n_steps=i_s - i_t, nfe_per_sample=i_s - i_t,
                          x_t_sha256=_digest(x_t.detach().cpu().numpy()), x_t_info=xt_info)
            np.savez_compressed(out / "samples" / f"{key}.npz", x0=x0,
                                x_t=x_t.detach().cpu().numpy(), x_s=raw, x_s_denoised=den,
                                rollout_seeds=np.asarray(seeds), t=tg, s=sg)
            condition_data[key] = dict(record=record, denoised=den)

    ordered = list(condition_data)
    all_images = np.concatenate([condition_data[k]["denoised"] for k in ordered])
    log(f"[exp1] extracting prespecified DINOv2 features for {len(all_images)} images")
    all_features = Fe.extract("dinov2_vitb14", all_images, device=dev, batch_size=bs)
    cuts = np.cumsum([0] + [len(condition_data[k]["denoised"]) for k in ordered])
    dg = cfg["diagnostics"]
    kw = dict(pca_dims=int(dg["pca_dims"]), max_k=int(dg["gmm_max_components"]),
              covariance_type=dg["gmm_covariance"], cv_folds=int(dg["cv_folds"]),
              n_boot=int(dg["n_boot"]), n_null=int(dg["n_null"]), seed=int(dg["seed"]))
    rows, diagnostics = [], {}
    for j, key in enumerate(ordered):
        dat = condition_data[key]
        feats = all_features[cuts[j]:cuts[j + 1]]
        diag = D.analyze(feats, **kw)
        dat["features"], dat["labels"] = feats, np.asarray(diag["stability"]["labels"])
        rows.append(_row(dat["record"], diag))
        diagnostics[key] = _strip(diag)
        np.savez_compressed(out / "embeddings" / f"{key}.npz", dino_features=feats,
                            pca_scores=diag["pca_scores"], cluster_labels=dat["labels"])
        log(D.summarize(diag, key)[1])

    _write_csv(rows, out / "numerical" / "condition_summary.csv")
    Pv.dump_json(dict(rows=rows, diagnostics=diagnostics, settings=kw),
                 out / "numerical" / "condition_summary.json")
    _figures(rows, condition_data, [condition_data[_key(image_indices[0], s)]["record"]["s"]
                                    for s in endpoints], image_indices, out, int(dg["seed"]))

    lines = ["# Experiment 1 — transition length and conditional multimodality", "",
             f"Ran {len(image_indices)} fixed noisy states × {len(endpoints)} endpoints × {n} paired rollouts.",
             "DINOv2 ViT-B/14 was fixed before analysis; negative BIC(2)−BIC(1) favors two components.", "",
             "| image | s | Δt | preferred K (BIC) | BIC(2)−BIC(1) | bootstrap ARI |", "|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['image_index']} | {r['s']:.4f} | {r['delta_t']:.4f} | {r['preferred_k_bic']} | "
              f"{r['delta_bic_2_minus_1']:.2f} | {r['bootstrap_ari']:.3f} |" for r in rows]
    lines += ["", "Figures: `figures/figure_1a_*`, `figure_1b_*`, and `figure_1c_*`."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    provenance = Pv.snapshot(cfg, dev)
    provenance.update(status="finished", runtime_s=time.time() - started,
                      teacher=teacher.describe(), dataset=ds.describe(), rows=rows)
    Pv.dump_json(provenance, out / "provenance.json")
    log(f"[exp1] complete -> {out}")
    return rows


def replot(cfg, out_dir=None):
    """Regenerate aggregate figures from completed numerical/embedding files only."""
    out = Path(out_dir or ROOT / cfg["output"]["dir"])
    payload = json.loads((out / "numerical" / "condition_summary.json").read_text())
    rows = payload["rows"]
    condition_data = {}
    for r in rows:
        key = _key(r["image_index"], r["s"])
        z = np.load(out / "embeddings" / f"{key}.npz")
        condition_data[key] = dict(features=z["dino_features"], labels=z["cluster_labels"])
    image_indices = [int(v) for v in cfg["experiment"]["image_indices"]]
    endpoints = sorted({r["s"] for r in rows}, reverse=True)
    _figures(rows, condition_data, endpoints, image_indices, out, int(cfg["diagnostics"]["seed"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="exp1_transition_multimodality")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args(argv)
    cfg = C.load_config(args.config)
    return replot(cfg, out_dir=args.out) if args.plot_only else run(
        cfg, out_dir=args.out, device=args.device)


if __name__ == "__main__":
    main()
