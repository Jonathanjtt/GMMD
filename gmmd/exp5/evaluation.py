"""exp5 evaluation stages (dispatched by :mod:`gmmd.experiments.ibw_amortized`).

* ``evalstudent``   stages A, B, C at held-out anchors, with matched sample counts:
                    A teacher -> IBW (an independent evaluation-only IBW fit at the anchor),
                    B IBW -> student (forward/reverse KL per dim, NLL, sample MMD vs an IBW floor),
                    C teacher -> student; plus component usage.
* ``endpoints``     cached fp32 teacher endpoints at training anchors (direct-NLL ablation data).
* ``generate``      complete samplers from x_T: the student (standalone sampler) and coarse teachers.
* ``fid``           FID / precision / recall at identical n, preprocessing and reference, with a
                    paired bootstrap against exp4's fp32 teacher images (same x_T seeds).
* ``timing``        uncached end-to-end generation after warm-up, device-synchronised.
* ``shift``         rollout shift: student boundary states vs teacher trajectory states.
* ``refine-anchors`` student-rollout states at the interval boundaries (refinement anchors).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gmmd import fid as FID
from gmmd.exp5 import common as C
from gmmd.exp5 import grid as G
from gmmd.exp5 import ibw_targets as IT
from gmmd.exp5 import metrics as MT
from gmmd.exp5.common import log
from gmmd.exp5.student import mixture_log_prob
from gmmd.exp5.train import endpoint_path

__all__ = ["STAGES"]


def _ckpt(cfg, tag, which="best"):
    return C.out_dir(cfg) / "student" / tag / f"ckpt_{which}.pt"


def _student(cfg, tag, which, device):
    from gmmd.exp5.sampler import StudentSampler
    return StudentSampler(_ckpt(cfg, tag, which), device=device, precision="tf32")


# ============================================================================== stage A/B/C
def stage_evalstudent(cfg, a, rest):
    ap = argparse.ArgumentParser(prog="evalstudent")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--fit-variant", default="frozen")
    o = ap.parse_args(rest)
    ce = cfg["evaluation"]
    root = C.out_dir(cfg) / "eval" / a.tag / a.split
    have = lambda n, j: (root / f"traj{n:03d}_j{j}.json").exists()
    import gmmd.experiments.ibw_amortized as D
    tasks = [(n, j) for n, j in D._tasks(cfg, a, a.split, have)
             if C.ref_path(cfg, a.split, n, j).exists()
             and C.fit_path(cfg, a.split, o.fit_variant, n, j).with_suffix(".json").exists()]
    log(f"evalstudent {a.tag}/{a.split} ({o.ckpt}): {len(tasks)} pairs")
    if not tasks:
        return
    teacher = C.build_teacher(cfg, a.device, tf32=False)
    parts = C.partition(cfg, teacher)
    bs = int(cfg["teacher"]["batch_size"])
    S = _student(cfg, a.tag, o.ckpt, a.device)
    net = S.net
    vid, _ = C.variant_cfg(cfg, o.fit_variant)
    nm, nkl, R = int(ce["n_model"]), int(ce["n_kl"]), int(ce["n_ref"])
    for n, j in tasks:
        iv = parts[j]
        x = C.load_anchor(cfg, a.split, n, j).to(a.device)[None]
        d = x[0].numel()
        with torch.inference_mode(), IT.tf32(True):
            logits, means, logv = net(x, j)
        logits, means, logv = logits[0].float(), means[0].reshape(net.K, -1).float(), logv[0].float()
        pi = torch.softmax(logits, -1)
        g = torch.Generator(device=a.device).manual_seed(G.seed_of(C.NS_EVAL, C.sid(a.split), n, j, 999))
        ys, ks = IT.sample_mixture(means, (logv.exp()), nm, g, weights=pi)
        with np.load(C.fit_path(cfg, a.split, o.fit_variant, n, j)) as z:
            fm = torch.as_tensor(z["means"]).to(a.device)
            fv = torch.as_tensor(z["variances"]).to(a.device).float()
        g2 = torch.Generator(device=a.device).manual_seed(G.seed_of(C.NS_EVAL, C.sid(a.split), n, j, vid, 0))
        zi, _ = IT.sample_mixture(fm, fv, 2 * nm, g2)
        yi = zi.reshape(2 * nm, *teacher.image_shape)
        if iv.terminal:
            with IT.tf32(False):
                yi, _ = G.terminal_denoise(teacher, yi, bs)
        rz = np.load(C.ref_path(cfg, a.split, n, j))
        refA, refB = (rz["dino"][:R], rz["pixel"][:R]), (rz["dino"][R:], rz["pixel"][R:])
        fS = MT.featurize(teacher, iv, ys.reshape(nm, *teacher.image_shape).cpu(), ce, bs, a.device)
        fI = MT.featurize(teacher, iv, yi.cpu(), ce, bs, a.device)
        fIA, fIB = (fI[0][:nm], fI[1][:nm]), (fI[0][nm:], fI[1][nm:])
        seed = n * 100 + j
        out = dict(split=a.split, n=n, j=j, tag=a.tag, ckpt=str(_ckpt(cfg, a.tag, o.ckpt)), step=S.step,
                   stage_A=MT.compare(refA, refB, fIA, ce, seed),
                   stage_B=MT.compare(fIA, fIB, fS, ce, seed),
                   stage_C=MT.compare(refA, refB, fS, ce, seed))
        # likelihood-based stage B (exact densities of both mixtures; standardised coordinates)
        sc = float(np.exp(0.5 * net.log_vref[j].item()))
        xf = x.reshape(1, -1)
        lw = torch.log_softmax(logits, -1)
        with torch.no_grad():
            gk = torch.Generator(device=a.device).manual_seed(G.seed_of(C.NS_EVAL, C.sid(a.split), n, j, 998))
            if not iv.terminal:
                zk, _ = IT.sample_mixture(fm, fv, nkl, gk)                         # z ~ q_IBW
                u = (zk - xf) / sc
                lq_ibw = IT.isotropic_log_prob(u, (fm - xf) / sc, fv / sc ** 2)
                lq_stu = mixture_log_prob(u[None].float(), ((means - xf) / sc)[None], (logv - 2 * np.log(sc))[None],
                                          logits[None])[0].double()
                yk, _ = IT.sample_mixture(means, logv.exp(), nkl, gk, weights=pi)  # y ~ q_phi
                v_ = (yk - xf) / sc
                rq_ibw = IT.isotropic_log_prob(v_, (fm - xf) / sc, fv / sc ** 2)
                rq_stu = mixture_log_prob(v_[None].float(), ((means - xf) / sc)[None], (logv - 2 * np.log(sc))[None],
                                          logits[None])[0].double()
                out["kl_forward_per_dim"] = float((lq_ibw - lq_stu).mean() / d)   # KL(q_IBW || q_phi)
                out["kl_reverse_per_dim"] = float((rq_stu - rq_ibw).mean() / d)   # KL(q_phi || q_IBW)
                out["nll_ibw_samples_per_dim_std"] = float(-lq_stu.mean() / d)
                out["ibw_entropy_per_dim_std"] = float(-lq_ibw.mean() / d)
            else:
                u = (yi.reshape(2 * nm, -1).float() - xf) / sc
                lq_stu = mixture_log_prob(u[None], ((means - xf) / sc)[None], (logv - 2 * np.log(sc))[None],
                                          logits[None])[0].double()
                out["nll_ibw_samples_per_dim_std"] = float(-lq_stu.mean() / d)
        out["student"] = dict(weights=pi.tolist(), weight_entropy=float(-(pi * pi.clamp_min(1e-30).log()).sum()),
                              weight_perplexity=float(torch.exp(-(pi * pi.clamp_min(1e-30).log()).sum())),
                              sampled_components=torch.bincount(ks, minlength=net.K).tolist(),
                              variances_over_vref=(logv.exp() / np.exp(net.log_vref[j].item())).tolist(),
                              stats=IT.component_stats(means, logv.exp()))
        out["ibw"] = dict(stats=IT.component_stats(fm, fv), variances=fv.tolist())
        C.dump(out, root / f"traj{n:03d}_j{j}.json")
        log(f"evalstudent {a.split} traj {n} j {j}: DINOv2 excess A {out['stage_A']['dinov2']['excess']:+.4f} "
            f"B {out['stage_B']['dinov2']['excess']:+.4f} C {out['stage_C']['dinov2']['excess']:+.4f}; "
            f"KL fwd {out.get('kl_forward_per_dim', float('nan')):.4f}; pi perplexity "
            f"{out['student']['weight_perplexity']:.2f}")
    summarize_eval(cfg, a.tag, a.split)


def summarize_eval(cfg, tag, split):
    root = C.out_dir(cfg) / "eval" / tag / split
    rows = [json.loads(p.read_text()) for p in sorted(root.glob("traj*_j*.json"))]
    out = dict(tag=tag, split=split, n_pairs=len(rows), per_interval={}, overall={})
    for j in sorted({r["j"] for r in rows}):
        rs = [r for r in rows if r["j"] == j]
        e = {st: MT.summarize_rows([r[st] for r in rs]) for st in ("stage_A", "stage_B", "stage_C")}
        for k in ("kl_forward_per_dim", "kl_reverse_per_dim", "nll_ibw_samples_per_dim_std"):
            v = [r[k] for r in rs if k in r]
            if v:
                e[k] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        e["student_weight_perplexity"] = float(np.mean([r["student"]["weight_perplexity"] for r in rs]))
        e["student_n_effective"] = float(np.mean([r["student"]["stats"]["n_effective"] for r in rs]))
        e["ibw_n_effective"] = float(np.mean([r["ibw"]["stats"]["n_effective"] for r in rs]))
        out["per_interval"][str(j)] = e
    out["overall"] = {st: MT.summarize_rows([r[st] for r in rows]) for st in ("stage_A", "stage_B", "stage_C")}
    C.dump(out, C.out_dir(cfg) / "eval" / tag / f"{split}_summary.json")
    return out


# ============================================================================== ablation data
def stage_endpoints(cfg, a, rest):
    ap = argparse.ArgumentParser(prog="endpoints")
    ap.add_argument("--n-endpoints", type=int, default=8)
    o = ap.parse_args(rest)
    import gmmd.experiments.ibw_amortized as D
    tasks = D._tasks(cfg, a, a.split, lambda n, j: endpoint_path(cfg, a.split, n, j).exists())
    log(f"endpoints {a.split} shard {a.shard}/{a.n_shards}: {len(tasks)} pairs x {o.n_endpoints}")
    if not tasks:
        return
    teacher = C.build_teacher(cfg, a.device, tf32=cfg["precision"]["teacher_reference_tf32"])
    parts = C.partition(cfg, teacher)
    bs = int(cfg["teacher"]["batch_size"])
    E = int(o.n_endpoints)
    per = max(1, bs // E)
    by_j = {}
    for n, j in tasks:
        by_j.setdefault(j, []).append(n)
    for j, ns in sorted(by_j.items()):
        pending = list(ns)
        while pending:
            grp = []
            while pending and len(grp) < per:
                n = pending.pop(0)
                if not endpoint_path(cfg, a.split, n, j).exists() and C.claim(C.lock_path(cfg, "endpoints", a.split, n, j)):
                    grp.append(n)
            if not grp:
                break
            try:
                _endpoint_group(cfg, a, teacher, parts, j, grp, E, bs)
            finally:
                for n in grp:
                    C.release(C.lock_path(cfg, "endpoints", a.split, n, j))


def _endpoint_group(cfg, a, teacher, parts, j, grp, E, bs):
    x = torch.stack([C.load_anchor(cfg, a.split, n, j) for n in grp])
    seeds = np.asarray([[G.seed_of(C.NS_ENDPOINT, C.sid(a.split), n, j, e) for e in range(E)] for n in grp])
    with IT.timed(a.device) as tm:
        Y, nfe = G.interval_rollouts(teacher, x, parts[j], seeds, batch_size=bs)
    for k, n in enumerate(grp):
        C.save_npz_atomic(endpoint_path(cfg, a.split, n, j), Y=Y[k].cpu().numpy(), seeds=seeds[k])
        C.dump(dict(split=a.split, n=n, j=j, n_endpoints=E, nfe=nfe // len(grp),
                    seconds=tm.seconds / len(grp), precision=C.env_precision()),
               endpoint_path(cfg, a.split, n, j).with_suffix(".json"))
    log(f"endpoints j {j} trajs {grp}: {tm.seconds:.0f}s")


# ============================================================================== generation
def _method_dir(cfg, method):
    return C.out_dir(cfg) / "generation" / method.replace(":", "_")


def _chunk_seeds(cfg, c):
    g = cfg["generation"]
    return [int(g["xT_seed_base"]) + c * int(g["chunk"]) + k for k in range(int(g["chunk"]))]


def stage_generate(cfg, a, rest):
    """``--method student:<tag>[:<ckpt>]`` or ``coarse:<kind>:<n_steps>``; ``--n`` images."""
    ap = argparse.ArgumentParser(prog="generate")
    ap.add_argument("--method", required=True)
    ap.add_argument("--n", type=int, default=None)
    o = ap.parse_args(rest)
    gc = cfg["generation"]
    n_chunks = int(o.n or gc["n_images"]) // int(gc["chunk"])
    root = _method_dir(cfg, o.method)
    root.mkdir(parents=True, exist_ok=True)
    todo = [c for c in range(n_chunks) if c % a.n_shards == a.shard and not (root / f"chunk{c:03d}.npz").exists()]
    log(f"generate {o.method}: chunks {todo}")
    if not todo:
        return
    kind, *args = o.method.split(":")
    dev = torch.device(a.device)
    if kind == "student":
        S = _student(cfg, args[0], args[1] if len(args) > 1 else "best", a.device)
    else:
        teacher = C.build_teacher(cfg, a.device, tf32=cfg["precision"]["teacher_reference_tf32"])
    for c in todo:
        seeds = _chunk_seeds(cfg, c)
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        extra = {}
        if kind == "student":
            out, rec = S.sample(seeds, batch_size=len(seeds), keep_intermediates=(c == 0))
            x0 = out["images"]
            extra["components"] = out["components"].numpy()
            extra["weight_entropy"] = out["weight_entropy"].numpy()
            if c == 0:
                extra["states"] = out["states"].half().numpy()          # (B, 11, C, H, W)
            nfe, teacher_nfe = rec["network_evals_per_image"] * len(seeds), 0
        else:
            x0, nfe = G.coarse_trajectory(teacher, seeds, int(args[1]), kind=args[0],
                                          batch_size=int(cfg["teacher"]["batch_size"]))
            x0 = x0.cpu()
            rec, teacher_nfe = dict(kind=args[0], n_steps=int(args[1])), nfe
        torch.cuda.synchronize(dev)
        secs = time.perf_counter() - t0
        rng = dict(min=float(x0.min()), max=float(x0.max()), frac_clipped=float(((x0 < 0) | (x0 > 1)).float().mean()),
                   finite=bool(torch.isfinite(x0).all()))
        C.save_npz_atomic(root / f"chunk{c:03d}.npz", images=FID.to_uint8_upstream(x0), seeds=np.asarray(seeds), **extra)
        C.dump(dict(method=o.method, chunk=c, n=len(seeds), seconds=secs, network_evals=nfe,
                    teacher_score_evals=teacher_nfe, peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                    range=rng, record=rec, precision=C.env_precision(), gpu=torch.cuda.get_device_name(dev)),
               root / f"chunk{c:03d}.json")
        log(f"generate {o.method} chunk {c}: {secs:.1f}s for {len(seeds)} images; range [{rng['min']:.3f}, {rng['max']:.3f}]")


def _images(cfg, method, n):
    if method == "teacher":
        d = C._REPO / cfg["generation"]["exp4_teacher_dir"]
    else:
        d = _method_dir(cfg, method)
    chunks = sorted(d.glob("chunk*.npz"))
    imgs, seeds = [], []
    for p in chunks:
        z = np.load(p)
        imgs.append(z["images"])
        seeds.append(z["seeds"])
        if sum(len(s) for s in seeds) >= n:
            break
    imgs, seeds = np.concatenate(imgs)[:n], np.concatenate(seeds)[:n]
    if len(imgs) < n:
        raise RuntimeError(f"{method}: only {len(imgs)} images, need {n}")
    return imgs, seeds


def stage_fid(cfg, a, rest):
    ap = argparse.ArgumentParser(prog="fid")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--n", type=int, required=True)
    o = ap.parse_args(rest)
    from gmmd.experiments.interval_replacement import reference_features
    from torch_fidelity.metric_fid import fid_features_to_statistics
    ref = reference_features(cfg, a.device)
    ref_stats = fid_features_to_statistics(torch.as_tensor(ref["fid"].astype(np.float64)))
    methods = ["teacher"] + [m for m in o.methods if m != "teacher"]
    imgs = {}
    for m in methods:
        imgs[m], s = _images(cfg, m, o.n)
        if m == "teacher":
            seeds0 = s
        elif not np.array_equal(s, seeds0):
            raise RuntimeError(f"{m}: x_T seeds differ from the teacher's; the paired bootstrap needs the same seeds")
    rng = np.random.default_rng(int(cfg["fid"]["pr_subset_seed"]))
    real_idx = rng.choice(len(ref["prc"]), o.n, replace=False)
    out = dict(tag=a.tag, n_images=o.n, n_reference=int(len(ref["fid"])), reference_splits=cfg["fid"]["reference_splits"],
               fid_extractor="inception-v3-compat/2048", pr_extractor="vgg16/fc2_relu", pr_k=int(cfg["fid"]["pr_k"]),
               preprocessing="sampler float output -> clip(x*255, 0, 255).astype(uint8), identical for every method",
               methods={})
    feats = {}
    for m in methods:
        f_fid = FID.extract(imgs[m], "fid", a.device)
        f_prc = FID.extract(imgs[m], "prc", a.device)
        feats[m] = f_fid
        out["methods"][m] = dict(fid=FID.fid(f_fid, ref_stats),
                                 **FID.precision_recall(f_prc, ref["prc"][real_idx], k=int(cfg["fid"]["pr_k"]), device=a.device))
        log(f"fid {a.tag} {m}: FID {out['methods'][m]['fid']:.2f} P {out['methods'][m]['precision']:.3f} "
            f"R {out['methods'][m]['recall']:.3f} (n = {o.n})")
    boot = FID.bootstrap_fid(feats, ref_stats, int(cfg["fid"]["n_boot"]), 0)
    for m in methods:
        out["methods"][m]["bootstrap"] = boot[m]
    f_real = ref["fid"][rng.choice(len(ref["fid"]), o.n, replace=False)]
    out["real_subset_fid_same_n"] = FID.fid(f_real, ref_stats)
    C.dump(out, C.out_dir(cfg) / f"fid_{a.tag}.json")


# ============================================================================== timing
def stage_timing(cfg, a, rest):
    """Uncached x_T -> x_0 for ``--n`` fresh images in one batch after a warm-up batch, synchronised."""
    ap = argparse.ArgumentParser(prog="timing")
    ap.add_argument("--method", required=True)        # student:<tag> | coarse:<kind>:<n> | teacher
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--precision", choices=["fp32", "tf32"], default="fp32")
    o = ap.parse_args(rest)
    dev = torch.device(a.device)
    seeds = [5_000_000 + k for k in range(o.n)]
    warm = [5_100_000 + k for k in range(min(8, o.n))]
    kind, *args = o.method.split(":")
    tf = o.precision == "tf32"
    if kind == "student":
        from gmmd.exp5.sampler import StudentSampler
        S = StudentSampler(_ckpt(cfg, args[0], args[1] if len(args) > 1 else "best"), device=a.device, precision=o.precision)
        run = lambda sd: S.sample(sd, batch_size=len(sd))[0]["images"]
    else:
        teacher = C.build_teacher(cfg, a.device, tf32=tf)
        bs = int(cfg["teacher"]["batch_size"])
        if kind == "coarse":
            run = lambda sd: G.coarse_trajectory(teacher, sd, int(args[1]), kind=args[0], batch_size=bs)[0]
        else:
            import gmmd.pipeline as P

            def run(sd):
                x = P.prior_sample(teacher, sd)
                _, x0, _ = P.fine_steps(teacher, x, 0, teacher.sde.N, G.SeededGenerators([G.seed_of(53, s) for s in sd], dev), bs)
                return x0
    if kind != "teacher":
        run(warm)
    else:
        _ = teacher.score(torch.zeros((8, *teacher.image_shape), device=dev), 0.5)   # warm the kernels
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    x0 = run(seeds)
    torch.cuda.synchronize(dev)
    secs = time.perf_counter() - t0
    rec = dict(method=o.method, precision=o.precision, n_images=o.n, batch=o.n, seconds=secs, s_per_image=secs / o.n,
               images_per_gpu_hour=3600 * o.n / secs, peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
               finite=bool(torch.isfinite(torch.as_tensor(x0)).all()), gpu=torch.cuda.get_device_name(dev),
               note="uncached x_T -> x_0, one batch, after a warm-up batch, device synchronised; GPU shared "
                    "with another user's light job")
    C.dump(rec, C.out_dir(cfg) / "timing" / f"{o.method.replace(':', '_')}_{o.precision}_n{o.n}.json")
    log(f"timing {o.method} ({o.precision}): {secs:.2f}s for {o.n} images = {secs / o.n:.4f} s/img, peak {rec['peak_gb']:.1f} GB")


# ============================================================================== rollout shift
def stage_shift(cfg, a, rest):
    """Boundary-state distributions: student rollouts (first generation chunk, with intermediates)
    vs teacher trajectories (train anchors), DINOv2 on Tweedie-denoised states + pixels, n = 32,
    with a teacher-vs-teacher floor; and the final images vs exp4's teacher images."""
    ap = argparse.ArgumentParser(prog="shift")
    ap.add_argument("--method", required=True)          # e.g. student:main
    o = ap.parse_args(rest)
    ce = cfg["evaluation"]
    teacher = C.build_teacher(cfg, a.device, tf32=False)
    parts = C.partition(cfg, teacher)
    bs = int(cfg["teacher"]["batch_size"])
    z = np.load(_method_dir(cfg, o.method) / "chunk000.npz")
    st = torch.as_tensor(z["states"]).float()                           # (64, 11, C, H, W)
    trajs = [n for n, _ in C.split_seeds(cfg, "train")][:64]
    out = dict(method=o.method, boundaries={})
    for j in range(1, len(parts)):
        iv = parts[j - 1]                                               # boundary b_j = end of interval j-1
        tA = torch.stack([C.load_anchor(cfg, "train", n, j) for n in trajs[:32]])
        tB = torch.stack([C.load_anchor(cfg, "train", n, j) for n in trajs[32:64]])
        s = st[:32, j]
        fA = MT.featurize(teacher, iv, tA, ce, bs, a.device)
        fB = MT.featurize(teacher, iv, tB, ce, bs, a.device)
        fS = MT.featurize(teacher, iv, s, ce, bs, a.device)
        row = MT.compare(fA, fB, fS, ce, seed=j)
        row.update(grid_index=parts[j].step_start, sigma=parts[j].sigma_start,
                   state_rms=dict(teacher=float(tA.pow(2).mean().sqrt()), student=float(s.pow(2).mean().sqrt())))
        out["boundaries"][str(j)] = row
        log(f"shift {o.method} boundary {j} (grid {parts[j].step_start}): DINOv2 excess {row['dinov2']['excess']:+.4f} "
            f"(p {row['dinov2']['p_model']:.3f}); pixel excess {row['pixel']['excess']:+.5f}")
    C.dump(out, C.out_dir(cfg) / "eval" / f"shift_{o.method.replace(':', '_')}.json")


def stage_refine_anchors(cfg, a, rest):
    """Student rollouts from the refine seeds; their states at b_0..b_9 become the anchors of
    split ``refine`` (IBW-fitted afterwards with the frozen configuration, intervals 1..9)."""
    ap = argparse.ArgumentParser(prog="refine-anchors")
    ap.add_argument("--ckpt", default="best")
    o = ap.parse_args(rest)
    S = _student(cfg, a.tag, o.ckpt, a.device)
    todo = [(n, s) for n, s in C.split_seeds(cfg, "refine") if not C.anchor_path(cfg, "refine", n).exists()]
    log(f"refine-anchors from student {a.tag}: {len(todo)} rollouts")
    for i in range(0, len(todo), 32):
        grp = todo[i:i + 32]
        out, rec = S.sample([s for _, s in grp], batch_size=len(grp), keep_intermediates=True)
        for k, (n, s) in enumerate(grp):
            C.save_npz_atomic(C.anchor_path(cfg, "refine", n), states=out["states"][k, :-1].numpy(),
                              interval_index=np.arange(S.n_intervals))
            C.dump(dict(split="refine", n=n, xT_seed=s, source=f"student {a.tag} ({o.ckpt}, step {S.step})",
                        draw_seed=G.seed_of(71, s), record=rec), C.anchor_path(cfg, "refine", n).with_suffix(".json"))


STAGES = {"evalstudent": stage_evalstudent, "endpoints": stage_endpoints, "generate": stage_generate,
          "fid": stage_fid, "timing": stage_timing, "shift": stage_shift, "refine-anchors": stage_refine_anchors}
