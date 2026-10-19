"""Post-hoc diagnostic of the terminal transition (interval 9 -> the final image): how much
per-pixel spread does each kernel leave around its own mean, from the same anchor?

    python -m gmmd.exp5.diag_terminal --split train --trajs 0 1 2 3 --rollouts 8 --students main refined

teacher   : ``rollouts`` fp32 200-step rollouts of interval 9 from the anchor, noise removal at the end
IBW       : the cached 32-row pushforward bank ``T(z)``, ``z ~ q_IBW`` (exactly what the student learns)
student   : ``rollouts`` draws of the student's last kernel at the same anchor (component + noise),
            and the drawn component's standard deviation

Per anchor: ``spread`` = RMS over pixels of the deviation from the kernel's own sample mean
(unbiased over samples), ``hf`` = RMS of the 3x3 Laplacian of that deviation, and ``bias`` = RMS
distance between the kernel's sample mean and the teacher's.  [0, 1] image units.
"""
import argparse

import numpy as np
import torch

from gmmd.config import load_config
from gmmd.exp5 import common as C
from gmmd.exp5 import grid as G
from gmmd.exp5 import ibw_targets as IT


def _lap(x):                                               # (n, C, H, W) -> (n, C, H-2, W-2)
    return (x[..., :-2, 1:-1] + x[..., 2:, 1:-1] + x[..., 1:-1, :-2] + x[..., 1:-1, 2:] - 4 * x[..., 1:-1, 1:-1])


def _summ(y, ref_mean=None):
    y = y.double()
    R = len(y)
    m = y.mean(0, keepdim=True)
    dev = (y - m) * np.sqrt(R / (R - 1))
    out = dict(spread=float(dev.pow(2).mean().sqrt()), hf=float(_lap(dev).pow(2).mean().sqrt()))
    if ref_mean is not None:
        out["bias"] = float((m[0] - ref_mean).pow(2).mean().sqrt())
    return out, m[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="exp5_ibw_amortized_10step_k8")
    ap.add_argument("--split", default="train")
    ap.add_argument("--trajs", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--students", nargs="*", default=["main", "refined"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, {})
    teacher = C.build_teacher(cfg, a.device, tf32=False)
    parts = C.partition(cfg, teacher)
    iv = parts[-1]
    assert iv.terminal
    from gmmd.exp5.evaluation import _student
    students = {t: _student(cfg, t, "best", a.device) for t in a.students}
    rows = []
    for n in a.trajs:
        x = C.load_anchor(cfg, a.split, n, iv.j)
        seeds = np.array([[G.seed_of(9_901, C.sid(a.split), n, r) for r in range(a.rollouts)]])
        with IT.tf32(False), torch.no_grad():
            yt, _ = G.interval_rollouts(teacher, x[None], iv, seeds, batch_size=a.rollouts)
        yt = yt[0].cpu()
        row = dict(n=n)
        row["teacher"], mt = _summ(yt)
        with np.load(C.fit_path(cfg, a.split, "frozen", n, iv.j)) as z:
            bank = torch.as_tensor(z["bank"]).reshape(-1, *teacher.image_shape)
            v_ibw = float(np.mean(z["variances"]))
        row["ibw_bank"], _ = _summ(bank, mt)
        row["ibw_fit_sd"] = float(np.sqrt(v_ibw))
        for t, S in students.items():
            with torch.inference_mode(), IT.tf32(True):
                logits, means, logv = S.net(x[None].to(a.device), iv.j)
            p = torch.softmax(logits[0].float(), -1)
            g = torch.Generator(device=a.device).manual_seed(G.seed_of(9_902, n))
            ys, ks = IT.sample_mixture(means[0].reshape(S.net.K, -1).float(), logv[0].float().exp(), a.rollouts, g,
                                       weights=p)
            ys = ys.reshape(a.rollouts, *teacher.image_shape).cpu()
            row[t], _ = _summ(ys, mt)
            row[t]["component_sd"] = float((p * logv[0].float().exp().sqrt()).sum())
        rows.append(row)
        print(f"traj {n}: " + " | ".join(f"{k} spread {v['spread']:.4f} hf {v['hf']:.4f}"
                                         + (f" bias {v['bias']:.4f}" if "bias" in v else "")
                                         for k, v in row.items() if isinstance(v, dict)), flush=True)
    keys = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    summary = {k: {s: float(np.mean([r[k][s] for r in rows])) for s in rows[0][k]} for k in keys}
    summary["ibw_fit_sd"] = float(np.mean([r["ibw_fit_sd"] for r in rows]))
    C.dump(dict(split=a.split, trajs=a.trajs, rollouts=a.rollouts, interval=G.partition_table([iv])[0],
                rows=rows, mean=summary), C.out_dir(cfg) / "figures" / "diag" / "terminal_spread.json")
    for k, v in summary.items():
        print(k, v if not isinstance(v, dict) else {s: round(x, 5) for s, x in v.items()})


if __name__ == "__main__":
    main()
