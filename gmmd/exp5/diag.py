"""Visual diagnostics of stage A: teacher interval samples next to IBW samples at chosen
fitting iterations, for one (split, trajectory, interval).  Non-terminal intervals are shown
as Tweedie-denoised states E[x_0 | y] (fp32), the terminal interval as final images
(IBW draws pushed through the teacher's noise removal, exactly as in stage A).

    python -m gmmd.exp5.diag --split val --variant prior_free --traj 0 --intervals 4 7 9 --its 25 300
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from gmmd.config import load_config
from gmmd.exp5 import common as C
from gmmd.exp5 import grid as G
from gmmd.exp5 import ibw_targets as IT


def _to_uint8(x):
    return (np.clip(np.asarray(x, np.float32).transpose(0, 2, 3, 1), 0, 1) * 255).round().astype(np.uint8)


def sample_rows(cfg, teacher, parts, split, variant, n, j, its, n_show, seed=0):
    """``{label: images (n_show, H, W, 3) uint8}``: teacher set A, then IBW at each iteration."""
    iv = parts[j]
    bs = int(cfg["teacher"]["batch_size"])
    rz = np.load(C.ref_path(cfg, split, n, j))
    rows = {"teacher": torch.as_tensor(rz["A"][:n_show])}
    path = C.fit_path(cfg, split, variant, n, j)
    with np.load(path) as z:
        for it in its:
            key = f"snap{it}_means" if f"snap{it}_means" in z.files else "means"
            m = torch.as_tensor(z[key]).to(teacher.device)
            v = torch.as_tensor(z[key.replace("means", "variances")]).to(teacher.device).float()
            g = torch.Generator(device=teacher.device).manual_seed(int(G.seed_of(91, n, j, it, seed)))
            y, _ = IT.sample_mixture(m, v, n_show, g)
            y = y.reshape(n_show, *teacher.image_shape)
            if iv.terminal:
                with IT.tf32(False):
                    y, _ = G.terminal_denoise(teacher, y, bs)
            rows[f"IBW {it} it"] = y.cpu()
    out = {}
    with IT.tf32(False):
        for k, y in rows.items():
            if not iv.terminal:
                y = teacher.tweedie_denoise(y.to(teacher.device), iv.t_end_state).cpu()
            out[k] = _to_uint8(y)
    return out


def save_rows(rows, path, pad=2):
    from PIL import Image, ImageDraw
    labels = list(rows)
    n, H, W, _ = rows[labels[0]].shape
    lw = 90
    canvas = np.full((len(labels) * (H + pad), lw + n * (W + pad), 3), 255, np.uint8)
    for r, k in enumerate(labels):
        for c in range(n):
            canvas[r * (H + pad):r * (H + pad) + H, lw + c * (W + pad):lw + c * (W + pad) + W] = rows[k][c]
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    for r, k in enumerate(labels):
        d.text((4, r * (H + pad) + H // 2 - 6), k, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--config", default="exp5_ibw_amortized_10step_k8")
    p.add_argument("--split", default="val")
    p.add_argument("--variant", default="prior_free")
    p.add_argument("--traj", type=int, default=0)
    p.add_argument("--intervals", type=int, nargs="+", default=[4, 7, 9])
    p.add_argument("--its", type=int, nargs="+", default=[25, 300])
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    teacher = C.build_teacher(cfg, a.device, tf32=False)
    parts = C.partition(cfg, teacher)
    for j in a.intervals:
        rows = sample_rows(cfg, teacher, parts, a.split, a.variant, a.traj, j, a.its, a.n_show)
        out = C.out_dir(cfg) / "figures" / "diag" / (
            f"samples_{a.split}{a.traj:03d}_j{j}_{a.variant}_it{'-'.join(map(str, a.its))}.png")
        print(save_rows(rows, out))


if __name__ == "__main__":
    main()
