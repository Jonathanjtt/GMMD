"""Stage B, training: amortise the cached IBW mixtures into the conditional mixture network.

    python -m gmmd.experiments.ibw_amortized train --tag pilot --steps 3000 --trajs 0 1 2 3 4 5 6 7
    python -m gmmd.experiments.ibw_amortized train --tag main                 # resumes if interrupted
    python -m gmmd.experiments.ibw_amortized train --tag ablation_endpoints --data endpoints

Objective (``--data ibw``, the main method): for (anchor, interval) pairs from the training
split and FRESH draws ``z ~ q_IBW(. | x, j)`` from the cached mixture every step,

    L(phi) = E_(x, j) E_z [ -log q_phi(z | x, j) ] = E KL(q_IBW || q_phi) + const,

the forward KL from the fitted IBW distributions to the student.  The terminal interval's target
is the pushforward ``T # q_IBW`` through the teacher's noise-removal map, available as a cached
bank of samples (``T`` needs the network, so fresh draws would mean teacher calls).  Nothing is
differentiated through IBW; Adam trains the network.

``--data endpoints`` is the ABLATION: the same network and loss on cached teacher endpoints
(rollouts of the interval from the same anchors), bypassing IBW.

The NLL is evaluated in standardised coordinates ``u = (z - x) / sqrt(v_ref,j)`` (a per-interval
constant shift of the NLL), so the ten intervals, whose scales span ten orders of magnitude,
contribute comparable gradients.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from gmmd.exp5 import common as C
from gmmd.exp5 import ibw_targets as IT
from gmmd.exp5.common import log
from gmmd.exp5.student import MixtureStudent, count_parameters, standardized_nll

__all__ = ["PairData", "stage_train", "student_dir", "reference_log_variances"]


def student_dir(cfg, tag):
    return C.out_dir(cfg) / "student" / tag


class PairData:
    """Every (anchor, interval) target of some splits, resident on the device.

    kind ``ibw``: ``means`` (K, d) + ``variances`` (K,), plus ``bank`` (M, d) for the terminal
    interval.  kind ``endpoints``: ``bank`` (E, d) of teacher endpoints for every interval."""

    def __init__(self, cfg, sources, n_intervals, device, kind="ibw"):
        self.kind = kind
        self.device = torch.device(device)
        self.pairs = []            # dicts: split, n, j, x, means, variances, bank
        for split, variant, trajs in sources:
            for n in trajs:
                for j in range(n_intervals):
                    rec = dict(split=split, n=n, j=j)
                    if kind == "ibw":
                        p = C.fit_path(cfg, split, variant, n, j)
                        if not p.with_suffix(".json").exists():
                            continue
                        with np.load(p) as z:
                            rec["means"] = torch.as_tensor(z["means"]).to(self.device)
                            rec["variances"] = torch.as_tensor(z["variances"]).float().to(self.device)
                            if "bank" in z.files:
                                rec["bank"] = torch.as_tensor(z["bank"]).to(self.device)
                        meta = json.loads(p.with_suffix(".json").read_text())
                        rec["terminal"] = bool(meta["interval"]["terminal"])
                        if rec["terminal"] and "bank" not in rec:
                            raise RuntimeError(f"{p}: terminal fit without its pushforward bank")
                    else:
                        p = endpoint_path(cfg, split, n, j)
                        if variant == "refs":          # validation endpoints = cached reference set A
                            p = C.ref_path(cfg, split, n, j)
                        if not p.exists():
                            continue
                        with np.load(p) as z:
                            Y = z["A"] if variant == "refs" else z["Y"]
                            rec["bank"] = torch.as_tensor(Y).reshape(len(Y), -1).to(self.device)
                        rec["terminal"] = j == n_intervals - 1
                    rec["x"] = C.load_anchor(cfg, split, n, j).to(self.device)
                    self.pairs.append(rec)
        self.by_j = {}
        for i, r in enumerate(self.pairs):
            self.by_j.setdefault(r["j"], []).append(i)
        self.intervals = sorted(self.by_j)

    def __len__(self):
        return len(self.pairs)

    def draw(self, idx, M, generator):
        """``M`` target samples for pair ``idx``: fresh IBW draws, or bank rows (with replacement)."""
        r = self.pairs[idx]
        if "bank" in r and (self.kind == "endpoints" or r["terminal"]):
            k = torch.randint(0, len(r["bank"]), (M,), device=self.device, generator=generator)
            return r["bank"][k]
        z, _ = IT.sample_mixture(r["means"], r["variances"], M, generator)
        return z

    def batch(self, B, M, generator, cpu_rng):
        """Balanced over intervals: an interval uniformly, then one of its pairs uniformly."""
        js = cpu_rng.choice(self.intervals, size=B)
        idx = [int(cpu_rng.choice(self.by_j[j])) for j in js]
        x = torch.stack([self.pairs[i]["x"] for i in idx])
        z = torch.stack([self.draw(i, M, generator) for i in idx])
        j = torch.as_tensor([self.pairs[i]["j"] for i in idx], device=self.device)
        return x, j, z, idx


def endpoint_path(cfg, split, n, j):
    return C.out_dir(cfg) / "endpoints" / split / f"traj{n:03d}_j{j}.npz"


def reference_log_variances(data, n_intervals):
    """``log v_ref,j``: median over the training pairs of the target's mean per-coordinate
    variance (the IBW variances; for a bank, its spread around its own mean)."""
    out = []
    for j in range(n_intervals):
        vals = []
        for i in data.by_j.get(j, []):
            r = data.pairs[i]
            if "bank" in r and (data.kind == "endpoints" or r["terminal"]):
                b = r["bank"].double()
                vals.append(float(b.var(0, unbiased=True).mean()))
            else:
                vals.append(float(r["variances"].double().mean()))
        if not vals:
            raise RuntimeError(f"no training pairs for interval {j}")
        out.append(math.log(float(np.median(vals))))
    return out


def _ema_update(ema, model, decay):
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)


@torch.no_grad()
def evaluate_pairs(student, data, log_vref, M, seed, batch=8):
    """Per-interval NLL (standardised, per dim) of fixed target draws; forward KL per dim where
    the target density is known (IBW mixtures, non-terminal)."""
    student.eval()
    g = torch.Generator(device=data.device).manual_seed(int(seed))
    per = {}
    for s in range(0, len(data), batch):
        idx = list(range(s, min(s + batch, len(data))))
        x = torch.stack([data.pairs[i]["x"] for i in idx])
        j = torch.as_tensor([data.pairs[i]["j"] for i in idx], device=data.device)
        z = torch.stack([data.draw(i, M, g) for i in idx])
        logits, means, logv = student(x, j)
        ls = 0.5 * log_vref[j]
        d = z[0].shape[-1]
        nll = standardized_nll(z, x.reshape(len(idx), -1), means.reshape(len(idx), student.K, -1),
                               logv, logits, ls)                                    # (b, M)
        for k, i in enumerate(idx):
            r = data.pairs[i]
            row = dict(nll=float(nll[k].mean()))
            if data.kind == "ibw" and not r["terminal"]:
                sc = ls[k].exp()
                u = (z[k] - x[k].reshape(1, -1)) / sc
                lq = IT.isotropic_log_prob(u, (r["means"] - x[k].reshape(1, -1)) / sc,
                                           r["variances"] / sc ** 2)
                row["kl"] = float((lq.float() / d + nll[k]).mean())                # E[log q_IBW - log q_phi]/d
            per.setdefault(r["j"], []).append(row)
    out = {}
    for j, rows in sorted(per.items()):
        out[str(j)] = dict(nll=float(np.mean([r["nll"] for r in rows])), n_pairs=len(rows))
        ks = [r["kl"] for r in rows if "kl" in r]
        if ks:
            out[str(j)]["kl"] = float(np.mean(ks))
    out["mean_nll"] = float(np.mean([v["nll"] for k, v in out.items() if k != "mean_nll"]))
    student.train()
    return out


def _parse(rest):
    ap = argparse.ArgumentParser(prog="train", allow_abbrev=False)
    ap.add_argument("--data", choices=["ibw", "endpoints"], default="ibw")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--train-trajs", type=int, nargs="*", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="extra sources split:variant (e.g. refine:frozen)")
    ap.add_argument("--init-from", default=None, help="fine-tune from this checkpoint (EMA weights)")
    ap.add_argument("--lr-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(rest)


def stage_train(cfg, a, rest):
    o = _parse(rest)
    cs = cfg["student"]
    dev = torch.device(a.device)
    tf = bool(cfg["precision"]["student_tf32"])
    torch.backends.cudnn.allow_tf32 = tf
    torch.backends.cuda.matmul.allow_tf32 = tf
    out = student_dir(cfg, a.tag)
    out.mkdir(parents=True, exist_ok=True)
    steps = int(o.steps or cs["main_steps"])
    nI = int(cfg["partition"]["n_intervals"])
    trajs = o.train_trajs if o.train_trajs is not None else [n for n, _ in C.split_seeds(cfg, "train")]
    sources = [("train", "frozen", trajs)]
    for e in o.extra:
        sp, var = e.split(":")
        sources.append((sp, var, sorted({int(p.stem[4:7]) for p in (C.out_dir(cfg) / "fits" / sp / var).glob("traj*_j*.json")})))
    train = PairData(cfg, sources, nI, dev, kind=o.data)
    # model selection uses the method's own kind of target: IBW fits for the main method; teacher
    # endpoints (the cached reference set A at the val anchors) for the ablation, so no IBW
    # information reaches the ablation at all
    val_trajs = [n for n, _ in C.split_seeds(cfg, "val")]
    val = (PairData(cfg, [("val", "frozen", val_trajs)], nI, dev, kind="ibw") if o.data == "ibw" else
           PairData(cfg, [("val", "refs", val_trajs)], nI, dev, kind="endpoints"))
    log(f"train {a.tag}: {len(train)} training pairs over intervals {train.intervals} from {[(s, v, len(t)) for s, v, t in sources]}; "
        f"{len(val)} val pairs; data={o.data}")

    teacher = C.build_teacher(cfg, dev, tf32=tf)
    parts = C.partition(cfg, teacher)
    ckpt_last = out / "ckpt_last.pt"
    if ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        intervals = ck["arch"]["intervals"]
        log(f"resuming {a.tag} from step {ck['step']}")
    else:
        ck = None
        lv = reference_log_variances(train, nI)
        intervals = [dict(j=iv.j, sigma_start=iv.sigma_start, mean_gap=iv.mean_gap, log_vref=lv[iv.j],
                          v_anc=iv.v_anc, terminal=iv.terminal) for iv in parts]
    if ck is None and o.init_from:
        init = torch.load(o.init_from, map_location="cpu", weights_only=False)
        student = MixtureStudent.from_arch(init["arch"])
        student.load_state_dict(init["ema"])
        intervals = init["arch"]["intervals"]
        log(f"fine-tuning from {o.init_from} (step {init.get('step')})")
    else:
        student = MixtureStudent.from_teacher(teacher.model, cfg["teacher"]["config"], intervals,
                                              n_components=int(cs["n_components"]), perturb=float(cs["perturb"]),
                                              seed=o.seed, head_width=int(cs["head_width"]),
                                              embed_dim=int(cs["embed_dim"]))
    meta = dict(sigma_max=float(teacher.sde.sigma_max), image_shape=list(teacher.image_shape),
                n_intervals=nI, teacher=C.teacher_id(teacher), partition=C.partition_table_dicts(parts),
                data=o.data, sources=[(s, v, list(t)) for s, v, t in sources], tag=a.tag,
                prior="x_T = sigma_max * N(0, I) from a CPU generator seeded by the image seed",
                draw_seed="seed_of(71, image seed)", precision=dict(train_tf32=tf),
                init_from=o.init_from, git=C.git_state())
    del teacher
    torch.cuda.empty_cache()
    student.to(dev).train()
    ema = MixtureStudent.from_arch(student.arch()).to(dev)
    ema.load_state_dict(student.state_dict())
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    head_names = {n for n, _ in student.named_parameters()
                  if n.startswith("head.") or n.startswith("embed.")
                  or any(n.startswith(f"net.all_modules.{i}.") for i in student._out_idx)}
    trunk = [p for n, p in student.named_parameters() if p.requires_grad and n not in head_names]
    headp = [p for n, p in student.named_parameters() if p.requires_grad and n in head_names]
    opt = torch.optim.AdamW([dict(params=trunk, lr=float(cs["lr"]) * o.lr_scale),
                             dict(params=headp, lr=float(cs["lr_head"]) * o.lr_scale)],
                            betas=(0.9, 0.999), weight_decay=float(cs["weight_decay"]))
    base_lrs = [g["lr"] for g in opt.param_groups]
    gen = torch.Generator(device=dev).manual_seed(int(o.seed) + 12345)
    cpu_rng = np.random.default_rng(int(o.seed) + 777)
    step, best = 0, float("inf")
    if ck is not None:
        student.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, best = int(ck["step"]), float(ck["best"])
        gen.set_state(ck["gen_state"])
        cpu_rng.bit_generator.state = ck["cpu_rng"]
    lv_t = torch.tensor([iv["log_vref"] for iv in intervals], device=dev)
    B, M = int(cs["batch_pairs"]), int(cs["samples_per_pair"])
    warm = int(cs["warmup"])
    curves = out / "curves.jsonl"
    log(f"student: {count_parameters(student)} parameters ({count_parameters(student, True)} trainable); "
        f"trunk lr {base_lrs[0]:.2e}, head lr {base_lrs[1]:.2e}; B={B} pairs x M={M} draws; steps {step}->{steps}")
    if step == 0:
        ev = evaluate_pairs(ema, val, lv_t, 16, seed=999)
        _append(curves, dict(step=0, val=ev))
        log(f"step 0 val NLL {ev['mean_nll']:.4f} " + " ".join(f"j{k}:{v['nll']:.3f}" for k, v in ev.items() if k != "mean_nll"))
    run_loss, run_j = [], {}
    t0 = time.perf_counter()
    while step < steps:
        frac = step / max(1, steps)
        mult = min(1.0, (step + 1) / max(1, warm)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g_, b in zip(opt.param_groups, base_lrs):
            g_["lr"] = b * mult
        x, j, z, _ = train.batch(B, M, gen, cpu_rng)
        logits, means, logv = student(x, j)
        nll = standardized_nll(z, x.reshape(B, -1), means.reshape(B, student.K, -1), logv, logits,
                               0.5 * lv_t[j])                                       # (B, M)
        loss = nll.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(student.parameters(), float(cs["grad_clip"])))
        opt.step()
        _ema_update(ema, student, float(cs["ema"]))
        step += 1
        run_loss.append(loss.item())
        for jj, v in zip(j.tolist(), nll.mean(1).tolist()):
            run_j.setdefault(jj, []).append(v)
        if step % int(cs["eval_every"]) == 0 or step == steps:
            ev = evaluate_pairs(ema, val, lv_t, 16, seed=999)
            with torch.no_grad():
                pi = torch.softmax(logits.float(), -1)
                ent = float((-(pi * pi.clamp_min(1e-30).log()).sum(-1)).mean())
            rec = dict(step=step, loss=float(np.mean(run_loss)), grad_norm=gn, lr=opt.param_groups[0]["lr"],
                       train_nll={str(k): float(np.mean(v)) for k, v in sorted(run_j.items())},
                       weight_entropy=ent, logv_offset=float((logv - lv_t[j][:, None]).mean()),
                       val=ev, seconds=time.perf_counter() - t0)
            _append(curves, rec)
            log(f"step {step}: loss {rec['loss']:.4f} | val NLL {ev['mean_nll']:.4f} | " +
                " ".join(f"j{k}:{v['nll']:.3f}" + (f"/kl{v['kl']:.3f}" if 'kl' in v else "")
                         for k, v in ev.items() if k != "mean_nll") + f" | H(pi) {ent:.2f} | {rec['seconds']:.0f}s")
            run_loss, run_j = [], {}
            state = dict(arch=student.arch(), meta=meta, step=step, ema=ema.state_dict())
            if ev["mean_nll"] < best:
                best = ev["mean_nll"]
                torch.save(dict(state, best=best, val=ev), out / "ckpt_best.pt")
            if step % int(cs["ckpt_every"]) == 0 or step == steps:
                tmp = out / "ckpt_last.tmp.pt"
                torch.save(dict(state, model=student.state_dict(), opt=opt.state_dict(), best=best,
                                gen_state=gen.get_state(), cpu_rng=cpu_rng.bit_generator.state), tmp)
                tmp.rename(ckpt_last)
    torch.save(dict(arch=student.arch(), meta=meta, step=step, ema=ema.state_dict(), best=best),
               out / "ckpt_final.pt")
    C.dump(dict(tag=a.tag, steps=step, best_val_nll=best, seconds=time.perf_counter() - t0,
                n_train_pairs=len(train), parameters=count_parameters(student),
                trainable=count_parameters(student, True), config=cs, meta=meta),
           out / "train_summary.json")
    log(f"train {a.tag} done: {step} steps, best val NLL {best:.4f}")


def _append(path, rec):
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
