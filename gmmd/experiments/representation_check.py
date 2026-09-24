"""Does the multimodality verdict depend on the feature representation?  (H1 follow-up)

The H1 experiment scores its conditional samples in whatever representation its config names.
The first runs used ``features.extractor = "pixel"`` -- 8x average-pooled pixels -- and found no
multimodality anywhere.  That is a claim about *pixels*, not about the conditional: two different
faces at the same pose and lighting sit close together in pixel space, so a pixel metric can miss
a categorical difference that is obvious to the eye.

This module re-scores an EXISTING run's saved samples under several representations, without
repeating a single teacher rollout, and renders what a 2-means split actually separates in each.
It is the control for the representation itself: if a representation reports structure that the
SHORT-transition control also shows, that structure is not evidence about transition length.

    python -m gmmd.experiments.representation_check results/conditional_multimodality

Writes ``representation_check.json`` and ``clusters_<rep>.png`` into the run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gmmd import diagnostics as D
from gmmd import features as Fe
from gmmd import provenance as Pv
from gmmd.plotting import savefig, tile

__all__ = ["run", "main"]

DEFAULT_REPRESENTATIONS = ("dinov2_vitb14", "pixel")
GROUPS = ("long", "near", "deterministic")


def _extractor_kwargs(name, device, batch_size, pixel_downsample):
    if name == "pixel":
        return dict(downsample=pixel_downsample)
    return dict(device=device, batch_size=batch_size)


def run(run_dir, representations=DEFAULT_REPRESENTATIONS, device="cpu", batch_size=16,
        pixel_downsample=8, pca_dims=10, max_k=4, covariance_type="diag", cv_folds=5,
        n_boot=200, n_null=100, seed=0, log=print):
    """Re-score ``run_dir``'s saved conditional samples under each representation."""
    run_dir = Path(run_dir)
    samples = np.load(run_dir / "samples.npz")
    # The diagnostics run on the teacher's own denoised view of each sample: a raw x_s at
    # sigma ~ 1 is mostly noise, and the denoiser is a deterministic map, so it cannot create
    # modes that are not already there -- it only changes the metric they are measured in.
    groups = {"long": samples["x_s_denoised"], "near": samples["x_near_denoised"],
              "deterministic": samples["x_det_denoised"]}
    log(f"[representation_check] {run_dir}: " + ", ".join(f"{g} n={len(X)}" for g, X in groups.items()))

    kw = dict(pca_dims=pca_dims, max_k=max_k, covariance_type=covariance_type,
              cv_folds=cv_folds, n_boot=n_boot, n_null=n_null, seed=seed)
    report, labels = {}, {}
    for rep in representations:
        ekw = _extractor_kwargs(rep, device, batch_size, pixel_downsample)
        log(f"[representation_check] --- {rep} ---")
        feats = {}
        for g, X in groups.items():
            feats[g] = Fe.extract(rep, X, **ekw)
            r = D.analyze(feats[g], **kw)
            report[f"{rep}/{g}"] = {k: v for k, v in r.items() if k != "pca_scores"}
            for line in D.summarize(r, f"{rep} {g}"):
                log(line)
        labels[rep] = _cluster_figure(rep, feats["long"], groups["long"], run_dir, pca_dims, seed, log)

    # How much do two representations agree about WHICH samples group together?  A low value
    # means they are picking up different structure, not the same structure more or less well.
    from sklearn.metrics import adjusted_rand_score
    agreement = {f"{a} vs {b}": float(adjusted_rand_score(labels[a], labels[b]))
                 for i, a in enumerate(representations) for b in representations[i + 1:]}
    for k, v in agreement.items():
        log(f"[representation_check] 2-means split agreement, {k}: ARI {v:.3f}")

    out = dict(run=str(run_dir), representations=list(representations), settings=kw,
               groups={k: _strip(v) for k, v in report.items()},
               cluster_labels={k: v.tolist() for k, v in labels.items()},
               split_agreement=agreement, provenance=Pv.snapshot(None, device))
    Pv.dump_json(out, run_dir / "representation_check.json")
    log(f"[representation_check] -> {run_dir / 'representation_check.json'}")
    return out


def _cluster_figure(rep, features, images, run_dir, pca_dims, seed, log):
    """Render the two halves of a 2-means split, so the split can be READ, not just scored."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    k = int(min(pca_dims, features.shape[0] - 1, features.shape[1]))
    scores = PCA(n_components=k, svd_solver="full", random_state=seed).fit_transform(features)
    lab = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(scores)
    log(f"[representation_check] {rep}: 2-means sizes {np.bincount(lab).tolist()}")

    fig, axes = plt.subplots(2, 1, figsize=(16, 4.6), constrained_layout=True)
    for c in (0, 1):
        idx = np.where(lab == c)[0]
        axes[c].imshow(tile(images[idx], ncols=16), interpolation="nearest")
        axes[c].set_title(f"{rep}: cluster {c}  (n = {len(idx)})", fontsize=10)
        axes[c].set_xticks([])
        axes[c].set_yticks([])
    savefig(fig, run_dir / f"clusters_{rep}")
    plt.close(fig)
    return lab


def _strip(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = _strip(v)
            v.pop("labels", None)
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gmmd.experiments.representation_check",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", help="a finished run directory containing samples.npz")
    ap.add_argument("--representations", default=",".join(DEFAULT_REPRESENTATIONS))
    ap.add_argument("--device", default="cpu", help="device for the neural feature extractors")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return run(a.run_dir, representations=tuple(a.representations.split(",")), device=a.device,
               batch_size=a.batch_size, n_boot=a.n_boot, n_null=a.n_null, seed=a.seed)


if __name__ == "__main__":
    main()
