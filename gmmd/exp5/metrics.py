"""Conditional-fidelity metrics shared by stages A (teacher -> IBW), B (IBW -> student) and
C (teacher -> student), with the exp4 protocol: unbiased MMD^2 with a Gaussian kernel whose
bandwidth is the median heuristic on the REFERENCE pool, permutation p-values, sliced
Wasserstein, in two representations:

* DINOv2 ViT-B/14 CLS features of the teacher's Tweedie-denoised state ``E[x_0 | y]`` at the
  end time of the interval (for the terminal interval the samples already are images, and are
  featurised directly);
* 8x average-pooled raw pixels of the samples themselves.

Every comparison has a same-size control from the reference distribution itself (teacher A vs
teacher B, or IBW A vs IBW B), and is reported as ``excess = MMD^2(model, A) - MMD^2(B, A)``.
A non-significant p-value is "no detectable difference at this n", never equality.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.distance import pdist

from gmmd import diagnostics as D
from gmmd import features as Fe
from gmmd.exp5.ibw_targets import tf32

__all__ = ["featurize", "compare", "summarize_rows"]


def featurize(teacher, iv, y, cfg_e, batch_size=64, device="cuda"):
    """``(dinov2 (n, 768), pixel (n, D))`` for samples ``y`` (n, C, H, W) of interval ``iv``'s end.

    Runs in full fp32 regardless of the caller's TF32 setting, so every method is featurised by
    the identical computation."""
    y = torch.as_tensor(y, dtype=torch.float32)
    with tf32(False):
        if iv.terminal:
            den = y.numpy()
        else:
            den = torch.cat([teacher.tweedie_denoise(y[i:i + batch_size].to(teacher.device),
                                                     iv.t_end_state).cpu()
                             for i in range(0, len(y), batch_size)]).numpy()
        fd = Fe.extract(cfg_e["feature"], den, device=device, batch_size=32)
    fp = Fe.extract("pixel", y.numpy(), downsample=int(cfg_e["pixel_downsample"]))
    return fd, fp


def compare(ref_a, ref_b, model, cfg_e, seed):
    """Excess MMD^2 etc. of ``model`` against reference set A, with B vs A as the floor.

    Each argument is ``(dinov2, pixel)``.  Bandwidths come from A U B (the reference pool only)."""
    out = {}
    for r, rep in enumerate(("dinov2", "pixel")):
        A, B, G = ref_a[r], ref_b[r], model[r]
        h = D.median_bandwidth(A, B, seed=seed)
        g = D.mmd_permutation_test(G, A, n_perm=int(cfg_e["n_perm"]), bandwidth=h, seed=seed)
        b = D.mmd_permutation_test(B, A, n_perm=int(cfg_e["n_perm"]), bandwidth=h, seed=seed)
        row = dict(bandwidth=h, mmd2_model=g["mmd2"], p_model=g["p_value"], mmd2_floor=b["mmd2"],
                   p_floor=b["p_value"], excess=g["mmd2"] - b["mmd2"])
        if rep == "dinov2":
            row["swd_model"] = D.sliced_wasserstein(G, A, int(cfg_e["n_projections"]), seed=seed)["mean"]
            row["swd_floor"] = D.sliced_wasserstein(B, A, int(cfg_e["n_projections"]), seed=seed)["mean"]
        # within-set dispersion (mean pairwise distance): is the model as diverse as the reference?
        row["dispersion_model"] = float(pdist(np.asarray(G, np.float64)).mean())
        row["dispersion_ref"] = float(pdist(np.asarray(A, np.float64)).mean())
        row["dispersion_floor"] = float(pdist(np.asarray(B, np.float64)).mean())
        out[rep] = row
    return out


def summarize_rows(rows, alpha=0.05):
    """Means / SDs / detection rates of a list of ``compare`` outputs."""
    if not rows:
        return {}
    out = {}
    for rep in ("dinov2", "pixel"):
        ex = np.array([r[rep]["excess"] for r in rows], float)
        out[rep] = dict(n=len(rows), excess_mean=float(ex.mean()),
                        excess_sd=float(ex.std(ddof=1)) if len(ex) > 1 else 0.0,
                        mmd2_model_mean=float(np.mean([r[rep]["mmd2_model"] for r in rows])),
                        mmd2_floor_mean=float(np.mean([r[rep]["mmd2_floor"] for r in rows])),
                        detect_rate=float(np.mean([r[rep]["p_model"] < alpha for r in rows])),
                        floor_detect_rate=float(np.mean([r[rep]["p_floor"] < alpha for r in rows])))
        if rep == "dinov2":
            out[rep]["swd_model_mean"] = float(np.mean([r[rep]["swd_model"] for r in rows]))
            out[rep]["swd_floor_mean"] = float(np.mean([r[rep]["swd_floor"] for r in rows]))
        if all("dispersion_model" in r[rep] for r in rows):
            out[rep]["dispersion_ratio_mean"] = float(np.mean([r[rep]["dispersion_model"] / r[rep]["dispersion_ref"]
                                                               for r in rows]))
            out[rep]["dispersion_floor_ratio_mean"] = float(np.mean([r[rep]["dispersion_floor"] / r[rep]["dispersion_ref"]
                                                                     for r in rows]))
    return out
