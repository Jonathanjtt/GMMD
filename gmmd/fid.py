"""FID and precision/recall for exp4, on torch-fidelity's standard feature extractors.

* FID: ``inception-v3-compat`` pool3 (2048-d), the TF-Inception port that reproduces the
  original FID numbers; statistics and the Frechet distance via torch-fidelity's own functions.
* Precision / recall (Kynkaanniemi et al. 2019): ``vgg16`` ``fc2_relu`` features, k = 3, as
  torch-fidelity's ``prc`` metric.  P/R compares equal-sized sets: the generated images against
  a fixed random subset of the real reference of the same size (seeded, identical across methods).

Both extractors take uint8 (N, 3, H, W) and resize internally, so real and generated images go
through exactly the same path: 256x256 uint8 RGB in, no crop.  Real images are the lossless
CelebA-HQ 256 PNGs from the parquet shards; generated ones are the sampler's output quantised
as upstream ``run_lib`` does, ``clip(x * 255, 0, 255).astype(uint8)``.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch

__all__ = ["to_uint8_upstream", "iter_reference", "extract", "fid", "precision_recall",
           "bootstrap_fid"]


def to_uint8_upstream(x):
    """float (N, 3, H, W) in ~[0, 1] -> uint8 (N, H, W, 3), as score_sde's run_lib saves samples."""
    a = x.detach().cpu().permute(0, 2, 3, 1).numpy() if torch.is_tensor(x) else np.asarray(x)
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


def iter_reference(root, splits, batch=256):
    """uint8 (n, 256, 256, 3) batches of every image in the given CelebA-HQ parquet splits."""
    import pyarrow.parquet as pq
    from PIL import Image

    buf = []
    for split in splits:
        for f in sorted(Path(root).glob(f"{split}-*.parquet")):
            pf = pq.ParquetFile(f)
            for g in range(pf.metadata.num_row_groups):
                for row in pf.read_row_group(g, columns=["image"]).column("image").to_pylist():
                    buf.append(np.asarray(Image.open(io.BytesIO(row["bytes"])).convert("RGB")))
                    if len(buf) == batch:
                        yield np.stack(buf)
                        buf = []
    if buf:
        yield np.stack(buf)


_EXTRACTORS = {}


def _extractor(name, layer, device):
    from torch_fidelity.registry import FEATURE_EXTRACTORS_REGISTRY

    key = (name, layer, str(device))
    if key not in _EXTRACTORS:
        fe = FEATURE_EXTRACTORS_REGISTRY[name](name, features_list=[layer])
        _EXTRACTORS[key] = fe.to(device).eval()
    return _EXTRACTORS[key]


def extract(images_u8_nhwc, kind, device="cuda", batch=128):
    """``kind`` in {"fid", "prc"} -> float32 features (N, D) for uint8 NHWC images."""
    name, layer = {"fid": ("inception-v3-compat", "2048"), "prc": ("vgg16", "fc2_relu")}[kind]
    fe = _extractor(name, layer, device)
    out = []
    with torch.no_grad():
        for j in range(0, len(images_u8_nhwc), batch):
            x = torch.as_tensor(np.ascontiguousarray(images_u8_nhwc[j:j + batch])).permute(0, 3, 1, 2)
            out.append(fe(x.to(device))[0].float().cpu())
    return torch.cat(out).numpy()


def _stats(f):
    from torch_fidelity.metric_fid import fid_features_to_statistics
    return fid_features_to_statistics(torch.as_tensor(np.asarray(f, np.float64)))


def fid(f_gen, ref_stats):
    from torch_fidelity.metric_fid import fid_statistics_to_metric
    return float(fid_statistics_to_metric(_stats(f_gen), ref_stats, verbose=False)["frechet_inception_distance"])


def precision_recall(f_gen, f_real, k=3, device="cuda"):
    from torch_fidelity.metric_prc import prc_features_to_metric
    r = prc_features_to_metric(torch.as_tensor(f_gen).to(device), torch.as_tensor(f_real).to(device),
                               prc_neighborhood=k, prc_batch_size=10000, save_cpu_ram=False,
                               verbose=False)
    return dict(precision=float(r["precision"]), recall=float(r["recall"]), f_score=float(r["f_score"]))


class _FastFID:
    """FID against a fixed reference via one symmetric eigensolve per call, on the GPU.

    ``tr sqrt(S_g S_r) = sum sqrt(eig(A S_g A))`` with ``A = S_r^{1/2}``: ``S_g S_r`` is similar
    to the symmetric PSD ``A S_g A``, so the eigenvalues agree.  torch-fidelity's general
    eigensolve of a 2048 x 2048 non-symmetric matrix costs seconds on the CPU; the bootstrap
    needs thousands.  The point estimates still come from torch-fidelity (:func:`fid`); this is
    checked against it in :func:`bootstrap_fid`.
    """

    def __init__(self, ref_stats, device):
        self.device = device
        self.mu_r = torch.as_tensor(ref_stats["mu"], dtype=torch.float64, device=device)
        S_r = torch.as_tensor(ref_stats["sigma"], dtype=torch.float64, device=device)
        w, V = torch.linalg.eigh(S_r)
        self.A = (V * w.clamp(min=0).sqrt()) @ V.T
        self.tr_r = torch.trace(S_r)

    def __call__(self, f):
        f = torch.as_tensor(f, dtype=torch.float64, device=self.device)
        mu = f.mean(0)
        S = torch.cov(f.T)
        ev = torch.linalg.eigvalsh(self.A @ S @ self.A).clamp(min=0)
        d = mu - self.mu_r
        return float(d @ d + torch.trace(S) + self.tr_r - 2 * ev.sqrt().sum())


def bootstrap_fid(feats_by_method, ref_stats, n_boot, seed, baseline="teacher", device="cuda", pairwise=False):
    """Paired bootstrap over image indices (methods share x_T, so rows are paired).

    Returns per method the bootstrap SD of FID and, against ``baseline``, the 95% interval of
    the FID difference.  FID at n ~ 1000 is biased upward; the bias is common to equal-n methods
    but the interval below is for the n-image estimator, not for FID at n -> infinity.
    ``pairwise=True`` also returns ``{"a - b": [lo, hi]}`` for every ordered pair a before b (the
    same replicates, so the intervals are paired too).
    """
    fast = _FastFID(ref_stats, device)
    for m, f in feats_by_method.items():
        a, b = fast(f), fid(f, ref_stats)
        if abs(a - b) > 1e-3 * max(1.0, abs(b)):
            raise AssertionError(f"fast FID {a} disagrees with torch-fidelity {b} for {m}")
    rng = np.random.default_rng(seed)
    n = len(next(iter(feats_by_method.values())))
    boots = {m: np.empty(n_boot) for m in feats_by_method}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for m, f in feats_by_method.items():
            boots[m][b] = fast(f[idx])
    out = {}
    for m in feats_by_method:
        r = dict(sd=float(boots[m].std(ddof=1)))
        if m != baseline and baseline in boots:
            d = boots[m] - boots[baseline]
            r[f"diff_vs_{baseline}_ci95"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
        out[m] = r
    if pairwise:
        ms = list(feats_by_method)
        pw = {f"{a} - {b}": [float(np.quantile(boots[a] - boots[b], q)) for q in (0.025, 0.975)]
              for i, a in enumerate(ms) for b in ms[i + 1:]}
        return out, pw
    return out
