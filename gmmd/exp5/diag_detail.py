"""Per-interval image detail of one-step kernels vs the teacher's 200-step kernel.

Tests the mechanism behind the IBW targets' bias: a reverse-KL fit with a narrow isotropic
family is mode-seeking, and high-density regions of an image prior are smooth images, so every
IBW transition should move the state toward lower contrast and less fine detail than the
teacher's own interval.  For each (anchor, interval) of a split, 32 end states of each kernel --
teacher rollouts (reference set A), the cached IBW target (frozen fit; terminal: its noise-removed
bank), and the network kernels of the given students -- are Tweedie-denoised by the fp32 teacher
at the interval's end time (terminal: already images) and summarised per image by

* contrast: SD over all pixels and channels ([0, 1] scale);
* detail: RMS of the 3x3 Laplacian (fine-scale energy), averaged over channels.

Ratios to the teacher's value on the same anchor isolate the kernel from the anchor.

    python -m gmmd.exp5.diag_detail --split val --students init_main main
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from gmmd.config import load_config
from gmmd.exp5 import common as C
from gmmd.exp5 import grid as G
from gmmd.exp5 import ibw_targets as IT

NS_DETAIL = 95


def detail_stats(x):
    """``x`` (n, C, H, W) float in data scale -> per-image (contrast, laplacian RMS) arrays."""
    x = x.float().clamp(0, 1)
    contrast = x.reshape(len(x), -1).std(1)
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device).view(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(x.reshape(-1, 1, *x.shape[2:]), k)
    detail = lap.reshape(len(x), -1).pow(2).mean(1).sqrt()
    return contrast.cpu().numpy(), detail.cpu().numpy()


def _denoise(teacher, iv, y, bs):
    if iv.terminal:
        return y
    with IT.tf32(False):
        return torch.cat([teacher.tweedie_denoise(y[i:i + bs].to(teacher.device), iv.t_end_state)
                          for i in range(0, len(y), bs)])


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--config", default="exp5_ibw_amortized_10step_k8")
    p.add_argument("--split", default="val")
    p.add_argument("--variant", default="frozen")
    p.add_argument("--students", nargs="*", default=["init_main", "main"])
    p.add_argument("--intervals", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    from gmmd.exp5.sampler import StudentSampler
    teacher = C.build_teacher(cfg, a.device, tf32=False)
    parts = C.partition(cfg, teacher)
    bs = int(cfg["teacher"]["batch_size"])
    nets = {t: StudentSampler(C.out_dir(cfg) / "student" / t / "ckpt_best.pt", device=a.device).net
            for t in a.students}
    rows = []
    for n, _ in C.split_seeds(cfg, a.split):
        for j in a.intervals:
            iv = parts[j]
            if not (C.ref_path(cfg, a.split, n, j).exists() and C.fit_path(cfg, a.split, a.variant, n, j).exists()):
                continue
            sets = {"teacher": torch.as_tensor(np.load(C.ref_path(cfg, a.split, n, j))["A"][:a.n])}
            with np.load(C.fit_path(cfg, a.split, a.variant, n, j)) as z:
                if iv.terminal:
                    sets["ibw"] = torch.as_tensor(z["bank"][:a.n]).reshape(-1, *teacher.image_shape)
                else:
                    g = torch.Generator(device=a.device).manual_seed(G.seed_of(NS_DETAIL, C.sid(a.split), n, j))
                    y, _ = IT.sample_mixture(torch.as_tensor(z["means"]).to(a.device),
                                             torch.as_tensor(z["variances"]).to(a.device).float(), a.n, g)
                    sets["ibw"] = y.reshape(a.n, *teacher.image_shape)
            x = C.load_anchor(cfg, a.split, n, j).to(a.device)[None]
            for t, net in nets.items():
                with torch.inference_mode(), IT.tf32(True):
                    logits, means, logv = net(x, j)
                g = torch.Generator(device=a.device).manual_seed(G.seed_of(NS_DETAIL, C.sid(a.split), n, j, 1))
                y, _ = IT.sample_mixture(means[0].reshape(net.K, -1).float(), logv[0].float().exp(), a.n, g,
                                         weights=torch.softmax(logits[0].float(), -1))
                sets[t] = y.reshape(a.n, *teacher.image_shape)
            row = dict(n=n, j=j)
            for k, y in sets.items():
                c, d = detail_stats(_denoise(teacher, iv, y.to(a.device), bs))
                row[k] = dict(contrast=float(c.mean()), detail=float(d.mean()))
            rows.append(row)
            print(f"{a.split} traj {n} j {j}: " + "  ".join(
                f"{k} {row[k]['contrast']:.3f}/{row[k]['detail']:.4f}" for k in sets), flush=True)
    kinds = [k for k in rows[0] if k not in ("n", "j")]
    summary = {}
    for j in sorted({r["j"] for r in rows}):
        rs = [r for r in rows if r["j"] == j]
        summary[str(j)] = {k: {s: dict(mean=float(np.mean([r[k][s] for r in rs])),
                                       ratio_to_teacher=float(np.mean([r[k][s] / r["teacher"][s] for r in rs])))
                               for s in ("contrast", "detail")} for k in kinds}
    out = dict(split=a.split, variant=a.variant, students=a.students, n_per_set=a.n,
               definition="Tweedie-denoised end states (fp32 teacher; terminal: images): contrast = per-image "
                          "pixel SD, detail = per-image RMS of the 3x3 Laplacian; ratio = mean over anchors of "
                          "kernel / teacher on the same anchor", per_interval=summary, rows=rows)
    path = C.out_dir(cfg) / "figures" / "diag" / f"detail_per_interval_{a.split}.json"
    path.write_text(json.dumps(out, indent=1))
    print(path)
    for j, e in summary.items():
        print(f"j {j}: " + "  ".join(f"{k} {e[k]['contrast']['ratio_to_teacher']:.2f}/{e[k]['detail']['ratio_to_teacher']:.2f}"
                                     for k in kinds))


if __name__ == "__main__":
    main()
