"""Post-hoc diagnostic: held-out likelihood of the TEACHER's own transitions.

    python -m gmmd.exp5.diag_nll --split test --students main refined ablation_endpoints init_main

For every (anchor, interval) pair of ``split`` with cached teacher references (the two independent
sets ``A`` and ``B`` of 32 fp32 200-step rollouts of the interval from the anchor, pooled: M = 64),
the per-dimension negative log-likelihood of those rollouts under

  student:<tag> : the student's mixture at the anchor (``ckpt_best``, EMA weights, TF32 forward as
                  in sampling; the log density in float64)
  ibw           : the cached IBW fit itself (``frozen``; uniform weights; non-terminal intervals --
                  the terminal target is the pushforward ``T # q_IBW``, whose density is unknown)
  oracle_iso    : ONE isotropic Gaussian fitted to one reference set (sample mean, mean per-coordinate
                  variance) and scored on the other, both ways -- a held-out yardstick for what a
                  single isotropic Gaussian can reach; estimating its mean from h = 32 rollouts costs
                  it ~ 0.5 / h = 0.016 nats per dimension (``oracle_mean_penalty``)

in coordinates standardised by the interval's anchor variance ``v_anc,j`` (``u = (z - x) /
sqrt(v_anc,j)``, a per-interval constant shift), so methods compare directly within an interval.
Also per kernel, per coordinate and relative to the teacher's variance ``tau`` (unbiased, from the M rollouts):
``var:<name>`` = total variance (within-component variance plus the weighted spread of the component
means) / tau; ``bias2:<name>`` = squared distance between the kernel's mean and the teacher's sample
mean, minus the sample mean's own noise tau / M, over tau; ``sep:<name>`` = the weight-weighted mean
distance between two component means, in component standard deviations (unused components of a
student do not count).
Forward-KL view of stage A (teacher -> IBW) and stage C (teacher -> student); the ablation is the
model trained for exactly this quantity (its checkpoint was selected on the VAL reference sets,
never on test).
"""
import argparse
import json
import math

import numpy as np
import torch

from gmmd.config import load_config
from gmmd.exp5 import common as C
from gmmd.exp5 import ibw_targets as IT
from gmmd.exp5.student import mixture_log_prob


def _iso_gauss_nll(train, test):
    """Per-dim NLL of ``test`` (M, d) under N(mean(train), v I), v = mean per-coordinate variance."""
    m = train.mean(0, keepdim=True)
    v = train.var(0, unbiased=True).mean()
    d = test.shape[1]
    q = ((test - m) ** 2).sum(-1)
    return (0.5 * (d * (math.log(2 * math.pi) + torch.log(v)) + q / v) / d)


def _spread(means, v, w, A):
    """Kernel vs the teacher's rollouts ``A`` (M, d), per coordinate and in units of the teacher's
    variance tau: (total variance / tau, squared mean bias / tau with the M-sample noise of the
    teacher's mean removed, pi-weighted mean distance between two component means in SDs)."""
    means, v, w = means.double(), v.double(), w.double()
    d, M = means.shape[1], len(A)
    tau = float(A.var(0, unbiased=True).mean())
    mbar = (w[:, None] * means).sum(0, keepdim=True)
    tot = float((w * v).sum() + (w * ((means - mbar) ** 2).sum(-1)).sum() / d)
    bias2 = float(((A.mean(0, keepdim=True) - mbar) ** 2).sum() / d) - tau / M
    D = torch.cdist(means[None], means[None])[0]
    W = w[:, None] * w[None, :]
    W.fill_diagonal_(0)
    sep = float((W * D).sum() / W.sum() / (w * v).sum().sqrt())
    return tot / tau, bias2 / tau, sep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="exp5_ibw_amortized_10step_k8")
    ap.add_argument("--split", default="test")
    ap.add_argument("--students", nargs="*", default=["main", "refined", "ablation_endpoints", "init_main"])
    ap.add_argument("--variant", default="frozen")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, {})
    teacher = C.build_teacher(cfg, a.device, tf32=False)
    parts = C.partition(cfg, teacher)
    del teacher
    from gmmd.exp5.evaluation import _student
    nets = {t: _student(cfg, t, "best", a.device).net for t in a.students}
    rows = []
    for n, _ in C.split_seeds(cfg, a.split):
        for iv in parts:
            rp = C.ref_path(cfg, a.split, n, iv.j)
            if not rp.exists():
                continue
            with np.load(rp) as z:
                A = torch.as_tensor(np.concatenate([z["A"], z["B"]])).reshape(len(z["A"]) + len(z["B"]), -1)
                A = A.to(a.device).double()
            x = C.load_anchor(cfg, a.split, n, iv.j)
            xd = x.reshape(1, -1).to(a.device).double()
            ls = 0.5 * math.log(iv.v_anc)
            u = (A - xd) / math.exp(ls)
            d = u.shape[1]
            row = dict(n=n, j=iv.j, M=len(A))
            for t, net in nets.items():
                with torch.inference_mode(), IT.tf32(True):
                    logits, means, logv = net(x[None].to(a.device), iv.j)
                mu = (means.reshape(1, net.K, -1).double() - xd[:, None]) / math.exp(ls)
                lp = mixture_log_prob(u[None], mu, logv.double() - 2 * ls, logits.double())[0]
                row[f"student:{t}"] = float((-lp / d).mean())
                row[f"var:student:{t}"], row[f"bias2:student:{t}"], row[f"sep:student:{t}"] = _spread(
                    means.reshape(net.K, -1), logv[0].double().exp(), torch.softmax(logits[0].double(), -1), A)
            fp = C.fit_path(cfg, a.split, a.variant, n, iv.j)
            if not iv.terminal and fp.with_suffix(".json").exists():
                with np.load(fp) as z:
                    m, v = torch.as_tensor(z["means"]).to(a.device).double(), torch.as_tensor(z["variances"]).double()
                lq = IT.isotropic_log_prob(u, (m - xd) / math.exp(ls), v / math.exp(2 * ls))
                row["ibw"] = float((-lq / d).mean())
                row["var:ibw"], row["bias2:ibw"], row["sep:ibw"] = _spread(
                    m, v.to(a.device), torch.full((len(m),), 1 / len(m), device=a.device, dtype=torch.float64), A)
            h = len(u) // 2
            row["oracle_iso"] = float(0.5 * (_iso_gauss_nll(u[:h], u[h:]).mean() + _iso_gauss_nll(u[h:], u[:h]).mean()))
            rows.append(row)
            print(f"traj {n} j{iv.j}: " + " ".join(f"{k} {v:.4f}" for k, v in row.items()
                                                   if isinstance(v, float) and not k.startswith(("var:", "bias2:", "sep:"))), flush=True)
    keys = sorted({k for r in rows for k in r if isinstance(r[k], float)})
    per_j = {}
    for j in sorted({r["j"] for r in rows}):
        rs = [r for r in rows if r["j"] == j]
        per_j[str(j)] = {k: dict(mean=float(np.mean([r[k] for r in rs if k in r])),
                                 se=float(np.std([r[k] for r in rs if k in r], ddof=1) / np.sqrt(sum(k in r for r in rs)))
                                 if sum(k in r for r in rs) > 1 else None,
                                 n_pairs=int(sum(k in r for r in rs)))
                         for k in keys if any(k in r for r in rs)}
    out = dict(split=a.split, variant=a.variant, units="nats per dimension, u = (z - x) / sqrt(v_anc,j)",
               reference="cached reference sets A and B pooled (2 x 32 fp32 teacher rollouts per pair)",
               oracle_mean_penalty=0.5 / (rows[0]["M"] // 2), n_pairs=len(rows),
               per_interval=per_j, rows=rows)
    C.dump(out, C.out_dir(cfg) / "figures" / "diag" / f"nll_teacher_endpoints_{a.split}.json")
    for pre in ("", "var:", "bias2:", "sep:"):
        ks = [k for k in keys if (k.startswith(pre) if pre else not k.startswith(("var:", "bias2:", "sep:")))]
        print(f"{pre or 'nll'}\n{'j':>3} " + " ".join(f"{k.split(':', 1)[-1] if pre else k:>26}" for k in ks))
        for j, v in per_j.items():
            print(f"{j:>3} " + " ".join(f"{v[k]['mean']:>18.4f} (n={v[k]['n_pairs']:2d})" if k in v else " " * 26
                                        for k in ks))


if __name__ == "__main__":
    main()
