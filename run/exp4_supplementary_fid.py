"""exp4 supplement: FID / P-R at every sample size the stopped run reached, per method.

The 1,024-image run was stopped by decision after the pilot (teacher 16 chunks, coarse 7,
gmmd 3).  The three-way comparison stays at n = 192 (metrics_pilot.json).  This adds, on the
same reference statistics and extractors:
  * teacher at n = 192 / 448 / 1024 and real-vs-real at the same n: how much small n inflates FID;
  * teacher vs coarse at n = 448, paired bootstrap of the FID difference.

    CUDA_VISIBLE_DEVICES=0 python run/exp4_supplementary_fid.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from gmmd import fid as FID
from gmmd import provenance as Pv
from gmmd.config import load_config
from gmmd.experiments.interval_replacement import reference_features
from gmmd.experiments.interval_validation import out_dir

cfg = load_config("exp4_interval_replacement")
gen = out_dir(cfg) / "generation"
ref = reference_features(cfg, "cuda")
from torch_fidelity.metric_fid import fid_features_to_statistics
ref_stats = fid_features_to_statistics(torch.as_tensor(ref["fid"].astype(np.float64)))


def images(m, n_chunks):
    return np.concatenate([np.load(gen / m / f"chunk{c:03d}.npz")["images"] for c in range(n_chunks)])


out = dict(n_reference=int(len(ref["fid"])), rows=[])
feats = {}
for m, sizes in (("teacher", (3, 7, 16)), ("coarse", (7,))):
    f_all = FID.extract(images(m, max(sizes)), "fid")
    p_all = FID.extract(images(m, max(sizes)), "prc")
    feats[m] = f_all
    for k in sizes:
        n = 64 * k
        rng = np.random.default_rng(int(cfg["fid"]["pr_subset_seed"]))
        real_idx = rng.choice(len(ref["prc"]), n, replace=False)
        real_fid_idx = rng.choice(len(ref["fid"]), n, replace=False)
        row = dict(method=m, n=n, fid=FID.fid(f_all[:n], ref_stats),
                   real_subset_fid_same_n=FID.fid(ref["fid"][real_fid_idx], ref_stats),
                   **FID.precision_recall(p_all[:n], ref["prc"][real_idx], k=int(cfg["fid"]["pr_k"])))
        out["rows"].append(row)
        print(row)
n = 448
boot = FID.bootstrap_fid({m: feats[m][:n] for m in ("teacher", "coarse")}, ref_stats,
                         int(cfg["fid"]["n_boot"]), 0)
out["paired_448"] = boot
print(boot)
out["provenance"] = Pv.snapshot(cfg, "cuda")
Pv.dump_json(out, out_dir(cfg) / "metrics_supplementary.json")
