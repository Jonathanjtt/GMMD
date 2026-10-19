"""Post-hoc precision check of the offline IBW targets: TF32 (as used) vs fp32 with identical seeds.

    python -m gmmd.experiments.ibw_amortized fit --split test --variant frozen_fp32 --trajs 0 1 \
        --set precision.ibw_tf32=false
    python -m gmmd.experiments.ibw_amortized evalfits --split test --variant frozen_fp32 --trajs 0 1
    python -m gmmd.experiments.ibw_amortized evalfits --split test --variant frozen --trajs 0 1
    python -m gmmd.exp5.diag_precision --split test

``frozen_fp32`` shares the init, noise and evaluation seeds of ``frozen``, so its components
correspond one to one and every difference is due to precision.  For scale, the same quantities
are reported for the independent replicate ``frozen_r1`` (fresh seeds) -- the spread of the target
itself -- and the stage-A excess of both precisions is paired by anchor.
"""
import argparse
import json

import numpy as np

from gmmd.config import load_config
from gmmd.exp5 import common as C


def _load(cfg, split, variant, n, j):
    p = C.fit_path(cfg, split, variant, n, j)
    if not p.with_suffix(".json").exists():
        return None
    with np.load(p) as z:
        return z["means"].astype(np.float64), z["variances"].astype(np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="exp5_ibw_amortized_10step_k8")
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, {})
    root = C.out_dir(cfg)
    rows = []
    for meta in sorted((root / "fits" / a.split / "frozen_fp32").glob("traj*_j*.json")):
        d = json.loads(meta.read_text())
        n, j = int(d["n"]), int(d["j"])
        f32, t32, rep = (_load(cfg, a.split, v, n, j) for v in ("frozen_fp32", "frozen", "frozen_r1"))
        if t32 is None:
            continue
        x = C.load_anchor(cfg, a.split, n, j).double().numpy().reshape(1, -1)
        (m32, v32), (mt, vt) = f32, t32
        sd = np.sqrt(vt.mean())
        row = dict(n=n, j=j, precision=d["precision"],
                   # component k of fp32 vs component k of TF32 (same init): RMS per pixel, in units of
                   # the component standard deviation, and relative to the distance moved from x
                   mean_rms_over_sd=float(np.sqrt(((m32 - mt) ** 2).mean()) / sd),
                   mean_diff_over_offset=float(np.linalg.norm(m32 - mt) / np.linalg.norm(mt - x)),
                   var_ratio=float((v32 / vt).mean()))
        if rep is not None:                       # fresh seeds: components do not correspond -> compare
            mr, vr = rep                          # the mixture means' centroids and the variances
            row.update(rep_centroid_rms_over_sd=float(np.sqrt(((mr.mean(0) - mt.mean(0)) ** 2).mean()) / sd),
                       fp32_centroid_rms_over_sd=float(np.sqrt(((m32.mean(0) - mt.mean(0)) ** 2).mean()) / sd),
                       rep_var_ratio=float((vr.mean() / vt.mean())))
        for v in ("frozen", "frozen_fp32"):
            e = root / "fits_eval" / a.split / v / f"traj{n:03d}_j{j}.json"
            if e.exists():
                s = json.loads(e.read_text())["snapshots"]
                last = s[max(s, key=int)]
                row[f"excess_{v}"] = dict(dinov2=last["dinov2"]["excess"], pixel=last["pixel"]["excess"])
        rows.append(row)
    out = dict(split=a.split, n_pairs=len(rows), rows=rows)
    paired = [r for r in rows if "excess_frozen" in r and "excess_frozen_fp32" in r]
    if paired:
        dd = np.array([r["excess_frozen_fp32"]["dinov2"] - r["excess_frozen"]["dinov2"] for r in paired])
        dp = np.array([r["excess_frozen_fp32"]["pixel"] - r["excess_frozen"]["pixel"] for r in paired])
        out["paired_excess_fp32_minus_tf32"] = dict(
            n=len(paired), dinov2_mean=float(dd.mean()), dinov2_sd=float(dd.std(ddof=1)) if len(dd) > 1 else None,
            pixel_mean=float(dp.mean()), pixel_sd=float(dp.std(ddof=1)) if len(dp) > 1 else None,
            dinov2_tf32_mean=float(np.mean([r["excess_frozen"]["dinov2"] for r in paired])),
            pixel_tf32_mean=float(np.mean([r["excess_frozen"]["pixel"] for r in paired])))
    C.dump(out, root / "figures" / "diag" / f"precision_check_{a.split}.json")
    print(f"{'pair':>8} {'mean RMS/sd':>12} {'|dm|/|m-x|':>11} {'v32/vtf':>8} {'centroid fp32':>14} "
          f"{'centroid rep':>13} {'DINO tf32 → fp32':>18} {'pixel tf32 → fp32':>20}")
    for r in rows:
        e0, e1 = r.get("excess_frozen"), r.get("excess_frozen_fp32")
        ex = (f"{e0['dinov2']:+.3f} → {e1['dinov2']:+.3f}" if e0 and e1 else "")
        px = (f"{e0['pixel']:+.4f} → {e1['pixel']:+.4f}" if e0 and e1 else "")
        print(f"  n{r['n']} j{r['j']}: {r['mean_rms_over_sd']:12.4f} {r['mean_diff_over_offset']:11.4f} "
              f"{r['var_ratio']:8.4f} {r.get('fp32_centroid_rms_over_sd', np.nan):14.4f} "
              f"{r.get('rep_centroid_rms_over_sd', np.nan):13.4f} {ex:>18} {px:>20}")
    if paired:
        print(json.dumps(out["paired_excess_fp32_minus_tf32"], indent=1))


if __name__ == "__main__":
    main()
