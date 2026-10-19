"""exp5 driver: amortized IBW-GMMD with ten 8-component transitions over the complete trajectory.

    M="python -m gmmd.experiments.ibw_amortized"
    $M partition                                        # partition.json + split manifests
    $M anchors  --split train --shard 0 --n-shards 2    # fp32 teacher trajectories (states at b_j)
    $M refs     --split val   --shard 0 --n-shards 2    # fp32 teacher reference rollouts + features
    $M fit      --split val   --variant principal_r3 --shard 0 --n-shards 2     # TF32 IBW fits
    $M evalfits --split val   --variant principal_r3    # fp32 stage-A metrics at every snapshot
    $M select                                           # pre-registered rule -> selection.json
    $M freeze   --split val                             # selected snapshot -> fits/val/frozen
    $M fit      --split train --variant frozen ...      # training targets with the frozen config
    $M train / evalstudent / generate / fid / timing / shift / refine-anchors / report
                                                        # (gmmd.exp5.train, gmmd.exp5.evaluation)

Every stage is resumable at the granularity of its output files and shards its work over
``--n-shards`` workers (one per GPU).  Precision is set per stage, never inherited: fp32 for
everything the teacher produces as a reference, TF32 for IBW fitting and the student.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gmmd.config import load_config, parse_overrides
from gmmd.exp5 import common as C
from gmmd.exp5 import grid as G
from gmmd.exp5 import ibw_targets as IT
from gmmd.exp5 import metrics as MT
from gmmd.exp5.common import log

__all__ = ["main"]


# ============================================================================== partition
def light_teacher(cfg):
    """The teacher's SDE and grid without the network (CPU): enough for times and sigmas."""
    from gmmd.teachers.score_sde.configs import get_config
    from gmmd.teachers.score_sde.teacher import ScoreSDETeacher, build_sde

    tcfg = get_config(cfg["teacher"]["config"])
    sde, eps = build_sde(tcfg)

    def no_network(x, t):
        raise RuntimeError("light teacher has no score network")

    shape = (tcfg.data.num_channels, tcfg.data.image_size, tcfg.data.image_size)
    return ScoreSDETeacher(sde, no_network, shape, device="cpu", eps=eps)


def stage_partition(cfg, a):
    T = light_teacher(cfg)
    parts = C.partition(cfg, T)
    root = C.out_dir(cfg)
    rows = G.partition_table(parts)
    C.dump(dict(n_grid=int(T.sde.N), eps=T.eps, sigma_min=float(T.sde.sigma_min),
                sigma_max=float(T.sde.sigma_max), intervals=rows,
                note="interval j = teacher steps [step_start, step_end); the IBW target is the state "
                     "at grid index grid_end_state; the terminal interval's output is the noise-"
                     "removed x_0 = T(state at grid N-1)"), root / "partition.json")
    for split in ("train", "val", "test"):
        C.dump(dict(split=split, trajectories=[dict(n=n, xT_seed=s, traj_noise_seed=G.seed_of(51, s))
                                               for n, s in C.split_seeds(cfg, split)],
                    rule="x_T seed = base + n; trajectory noise seed = seed_of(51, x_T seed); splits "
                         "use disjoint x_T seed ranges, fixed before any target was generated"),
               root / "manifests" / f"split_{split}.json")
    lines = ["| j | teacher steps | grid start → end state | t start → end | σ start → end | gap σ_t²−σ_s² | v_anc | mean gap |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        end = f"{r['grid_end_state']} (+ noise removal)" if r["terminal"] else f"{r['grid_end_state']}"
        lines.append(f"| {r['j']} | [{r['step_start']}, {r['step_end']}) = {r['n_steps']} | {r['step_start']} → {end} "
                     f"| {r['t_start']:.5f} → {r['t_end_state']:.5f} | {r['sigma_start']:.4g} → "
                     f"{r['sigma_end_state']:.4g}{' → 0' if r['terminal'] else ''} | {r['gap']:.4g} "
                     f"| {r['v_anc']:.4g} | {r['mean_gap']:.4g} |")
    (root / "partition.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ============================================================================== anchors
def stage_anchors(cfg, a):
    root = C.out_dir(cfg) / "anchors" / a.split
    todo = [(n, s) for n, s in C.split_seeds(cfg, a.split)
            if n % a.n_shards == a.shard and not C.anchor_path(cfg, a.split, n).exists()]
    log(f"anchors {a.split} shard {a.shard}/{a.n_shards}: {len(todo)} trajectories")
    if not todo:
        return
    teacher = C.build_teacher(cfg, a.device, tf32=cfg["precision"]["teacher_reference_tf32"])
    parts = C.partition(cfg, teacher)
    bs = int(cfg["teacher"]["batch_size"])
    for i in range(0, len(todo), a.group):
        grp = todo[i:i + a.group]
        seeds = [s for _, s in grp]
        with IT.timed(a.device) as tm:
            states, nfe = G.trajectory_states(teacher, seeds, parts, batch_size=bs)
        for k, (n, s) in enumerate(grp):
            C.save_npz_atomic(C.anchor_path(cfg, a.split, n), states=states[:, k].numpy(),
                              grid_index=np.array([iv.step_start for iv in parts]))
            C.dump(dict(split=a.split, n=n, xT_seed=s, traj_noise_seed=G.seed_of(51, s),
                        nfe=nfe // len(grp), seconds=tm.seconds / len(grp), group_size=len(grp),
                        teacher=C.teacher_id(teacher), precision=C.env_precision(),
                        finite=bool(torch.isfinite(states[:, k]).all())),
                   C.anchor_path(cfg, a.split, n).with_suffix(".json"))
        log(f"anchors {a.split}: trajectories {[n for n, _ in grp]} in {tm.seconds:.0f}s ({nfe} NFE)")


def _tasks(cfg, a, split, have):
    """(n, j) pairs of this shard, filtered by --trajs / --intervals, skipping finished ones."""
    trajs = a.trajs if a.trajs else [n for n, _ in C.split_seeds(cfg, split)]
    js = a.intervals if a.intervals else list(range(int(cfg["partition"]["n_intervals"])))
    out = [(n, j) for n in trajs for j in js if not have(n, j)]
    return [t for k, t in enumerate(out) if k % a.n_shards == a.shard]


# ============================================================================== references
def stage_refs(cfg, a):
    """Teacher reference rollouts (sets A and B, R each) at (trajectory, interval) pairs, fp32,
    with their evaluation features."""
    ce = cfg["evaluation"]
    R = int(ce["n_ref"])
    tasks = _tasks(cfg, a, a.split, lambda n, j: C.ref_path(cfg, a.split, n, j).exists())
    log(f"refs {a.split} shard {a.shard}/{a.n_shards}: {len(tasks)} (trajectory, interval) pairs")
    if not tasks:
        return
    teacher = C.build_teacher(cfg, a.device, tf32=cfg["precision"]["teacher_reference_tf32"])
    parts = C.partition(cfg, teacher)
    bs = int(cfg["teacher"]["batch_size"])
    for n, j in tasks:
        lk = C.lock_path(cfg, "refs", a.split, n, j)
        if C.ref_path(cfg, a.split, n, j).exists() or not C.claim(lk):
            continue
        try:
            _ref_one(cfg, a, teacher, parts, n, j, R, bs, ce)
        finally:
            C.release(lk)


def _ref_one(cfg, a, teacher, parts, n, j, R, bs, ce):
    iv = parts[j]
    x0 = C.load_anchor(cfg, a.split, n, j)[None]
    seeds = C.ref_seeds(a.split, n, j, 2 * R)
    with IT.timed(a.device) as tm:
        ys, nfe = G.interval_rollouts(teacher, x0, iv, np.asarray(seeds)[None], batch_size=bs)
    ys = ys[0].cpu()
    with IT.timed(a.device) as tf:
        fd, fp = MT.featurize(teacher, iv, ys, ce, batch_size=bs, device=a.device)
    C.save_npz_atomic(C.ref_path(cfg, a.split, n, j), A=ys[:R].numpy(), B=ys[R:].numpy(),
                      seeds=np.asarray(seeds), dino=fd, pixel=fp)
    C.dump(dict(split=a.split, n=n, j=j, n_per_set=R, nfe=nfe, seconds=tm.seconds,
                feature_seconds=tf.seconds, interval=G.partition_table([iv])[0],
                teacher=C.teacher_id(teacher), precision=C.env_precision(),
                finite=bool(torch.isfinite(ys).all())),
           C.ref_path(cfg, a.split, n, j).with_suffix(".json"))
    log(f"refs {a.split} traj {n} j {j}: {2 * R} rollouts x {iv.n_steps} steps in {tm.seconds:.0f}s")


# ============================================================================== IBW fits
def _convergence(diag, cfg_c, stopped):
    """Per anchor: converged iff the last window's mean drift and log-variance drift are small."""
    it = diag["iterations"]
    v = np.log(np.maximum(diag["variances"], 1e-300))                    # (n_rec, A, K)
    md = diag["mean_drift"]                                               # (n_rec, A)
    out = []
    for k in range(md.shape[1]):
        dv = np.abs(v[1:, k] - v[:-1, k]).max(-1)                         # (n_rec-1,)
        ok = (md[1:, k] < float(cfg_c["mean_drift_tol"])) & (dv < float(cfg_c["logvar_drift_tol"]))
        first = None
        for r in range(len(ok)):                                          # first record from which ok holds to the end
            if ok[r:].all():
                first = int(it[r + 1])
                break
        out.append(dict(converged=bool(ok[-1]) and stopped[k] is None,
                        converged_from=first, final_mean_drift=float(md[-1, k]),
                        final_logvar_drift=float(dv[-1]) if len(dv) else None,
                        mean_drift_trace=md[1:, k].tolist(), logvar_drift_trace=dv.tolist()))
    return out


def _variant_settings(cfg, name, split):
    vid, v = C.variant_cfg(cfg, name)
    ci = cfg["ibw"]
    step_c = float(v.get("step_c", ci["step_c"]))
    iters = int(v.get("n_iterations", ci["max_iterations"]))
    if name == "frozen" or split in ("train", "test", "refine"):
        snaps = ()
    else:
        snaps = tuple(int(s) for s in ci["snapshot_iterations"] if int(s) <= iters)
    return vid, v, step_c, iters, snaps


def stage_fit(cfg, a):
    ci = cfg["ibw"]
    K, Bg = int(ci["n_components"]), int(ci["n_grad_samples"])
    vid, v, step_c, iters, snaps = _variant_settings(cfg, a.variant, a.split)
    if a.iterations:
        iters = int(a.iterations)
        snaps = tuple(s for s in snaps if s <= iters)
    want_bank = a.split in ("train", "test", "refine") or a.variant == "frozen"
    tasks = _tasks(cfg, a, a.split,
                   lambda n, j: C.fit_path(cfg, a.split, a.variant, n, j).with_suffix(".json").exists())
    log(f"fit {a.split}/{a.variant} shard {a.shard}/{a.n_shards}: {len(tasks)} fits; K={K} B={Bg} "
        f"step_c={step_c} iterations={iters} snapshots={list(snaps)} init={v}")
    if not tasks:
        return
    teacher = C.build_teacher(cfg, a.device, tf32=cfg["precision"]["ibw_tf32"])
    parts = C.partition(cfg, teacher)
    bs = int(cfg["teacher"]["batch_size"])
    by_j = {}
    for n, j in tasks:
        by_j.setdefault(j, []).append(n)
    done = lambda n, j: C.fit_path(cfg, a.split, a.variant, n, j).with_suffix(".json").exists()
    for j in sorted(by_j):
        iv = parts[j]
        pending = list(by_j[j])
        while pending:
            grp = []
            while pending and len(grp) < int(ci["anchor_batch"]):
                n = pending.pop(0)
                if not done(n, j) and C.claim(C.lock_path(cfg, "fit", a.split, a.variant, n, j)):
                    grp.append(n)
            if not grp:
                break
            try:
                _fit_group(cfg, a, teacher, iv, grp, vid, v, step_c, iters, snaps, K, Bg, bs, want_bank)
            finally:
                for n in grp:
                    C.release(C.lock_path(cfg, "fit", a.split, a.variant, n, j))


def _fit_group(cfg, a, teacher, iv, grp, vid, v, step_c, iters, snaps, K, Bg, bs, want_bank):
    ci = cfg["ibw"]
    s_id = C.sid(a.split)
    x = torch.stack([C.load_anchor(cfg, a.split, n, iv.j) for n in grp]).to(teacher.device)
    init_seeds = [G.seed_of(C.NS_INIT, s_id, n, iv.j, vid) for n in grp]
    noise_seeds = [G.seed_of(C.NS_NOISE, s_id, n, iv.j, vid) for n in grp]
    t0 = time.perf_counter()
    principal, pr_info = None, {}
    if v["scheme"] == "principal":
        pc = cfg["principal"]
        with IT.tf32(False), IT.timed(a.device) as tp:
            lam, E, nfe_p = IT.posterior_principal_directions(
                teacher, iv, x, q=K // 2, n_iter=int(pc["n_iter"]), oversample=int(pc["oversample"]),
                rel_h=float(pc["rel_h"]), seeds=[G.seed_of(C.NS_PRINCIPAL, s_id, n, iv.j) for n in grp],
                batch_size=bs)
        principal = (lam, E)
        pr_info = dict(nfe=nfe_p, seconds=tp.seconds, precision="fp32 (TF32 off)",
                       eigvals=lam.double().cpu().numpy().tolist(),
                       eig_over_gap=(lam.double().cpu().numpy() / iv.gap).tolist(), **pc)
    with IT.timed(a.device) as ti:
        m0, v0, info = IT.initial_mixture(v["scheme"], teacher, iv, x, K, init_seeds,
                                          spread=float(v["spread"]), var=v["var"], batch_size=bs,
                                          principal=principal)
    init_stats = [IT.component_stats(m0[k], v0[k]) for k in range(len(grp))]
    init_dist = (m0 - x.reshape(len(grp), 1, -1)).norm(dim=-1).double().cpu().numpy()
    with IT.timed(a.device) as tf:
        res = IT.fit_interval(teacher, iv, x, m0, v0, noise_seeds, step_c=step_c, n_iterations=iters,
                              n_grad_samples=Bg, snapshot_at=snaps, diag_every=int(ci["diag_every"]),
                              batch_size=bs)
    conv = _convergence(res["diagnostics"], ci["convergence"], res["stopped_early"])
    nfe_fit = res["nfe"]
    banks, bank_nfe, bank_s = [None] * len(grp), 0, 0.0
    if want_bank and iv.terminal:
        M = int(ci["terminal_bank"])
        with IT.timed(a.device) as tb:
            for k, n in enumerate(grp):
                g = torch.Generator(device=teacher.device).manual_seed(G.seed_of(C.NS_BANK, s_id, n, iv.j, vid))
                z, _ = IT.sample_mixture(res["means"][k], res["variances"][k], M, g)
                y, nb = G.terminal_denoise(teacher, z.reshape(M, *teacher.image_shape), bs)
                banks[k] = y.reshape(M, -1).float().cpu().numpy()
                bank_nfe += nb
        bank_s = tb.seconds
    total_s = time.perf_counter() - t0
    for k, n in enumerate(grp):
        arrays = dict(means=res["means"][k].float().cpu().numpy(),
                      variances=res["variances"][k].double().cpu().numpy(),
                      grad_norms=res["grad_norms"][:, k], diag_iterations=res["diagnostics"]["iterations"],
                      diag_variances=res["diagnostics"]["variances"][:, k],
                      diag_mean_drift=res["diagnostics"]["mean_drift"][:, k])
        for it, (mm, vv) in res["snapshots"].items():
            arrays[f"snap{it}_means"] = mm[k].float().cpu().numpy()
            arrays[f"snap{it}_variances"] = vv[k].double().cpu().numpy()
        if banks[k] is not None:
            arrays["bank"] = banks[k]
        path = C.fit_path(cfg, a.split, a.variant, n, iv.j)
        C.save_npz_atomic(path, **arrays)
        stats_final = IT.component_stats(res["means"][k], res["variances"][k])
        snap_stats = {str(it): IT.component_stats(mm[k], vv[k]) for it, (mm, vv) in res["snapshots"].items()}
        anchor_meta = json.loads(C.anchor_path(cfg, a.split, n).with_suffix(".json").read_text())
        meta = dict(
            split=a.split, n=n, xT_seed=anchor_meta["xT_seed"], j=iv.j, interval=G.partition_table([iv])[0],
            variant=a.variant, variant_cfg=v, n_components=K, n_grad_samples=Bg, step_c=step_c,
            step_size=step_c * iv.v_anc, n_iterations=iters, snapshots=list(snaps),
            init=dict(info, seeds=init_seeds[k], principal=pr_info and dict(
                pr_info, eigvals=pr_info["eigvals"][k], eig_over_gap=pr_info["eig_over_gap"][k]),
                stats=init_stats[k], dist_to_x=init_dist[k].tolist()),
            noise_seed=noise_seeds[k], stopped_early=res["stopped_early"][k], convergence=conv[k],
            grad_norm_first=float(res["grad_norms"][0, k]), grad_norm_last=float(res["grad_norms"][-1, k]),
            component_stats=stats_final, snapshot_stats=snap_stats,
            variances_over_v_anc=(res["variances"][k].double().cpu().numpy() / iv.v_anc).tolist(),
            mean_dist_to_x=(res["means"][k] - x[k].reshape(1, -1)).norm(dim=-1).double().cpu().numpy().tolist(),
            finite=bool(torch.isfinite(res["means"][k]).all() and torch.isfinite(res["variances"][k]).all()),
            bank=dict(n=int(ci["terminal_bank"]), nfe=bank_nfe // len(grp), seconds=bank_s / len(grp),
                      seed=G.seed_of(C.NS_BANK, C.sid(a.split), n, iv.j, vid)) if banks[k] is not None else None,
            cost=dict(group_size=len(grp), fit_nfe=nfe_fit // len(grp), fit_seconds=tf.seconds / len(grp),
                      init_nfe=(info["nfe"] + pr_info.get("nfe", 0)) // len(grp),
                      init_seconds=(ti.seconds + pr_info.get("seconds", 0.0)) / len(grp),
                      total_seconds=total_s / len(grp)),
            target=res["target"], teacher=C.teacher_id(teacher), precision=C.env_precision(),
            git=C.git_state())
        C.dump(meta, path.with_suffix(".json"))
    log(f"fit {a.split}/{a.variant} j={iv.j} trajs {grp}: {tf.seconds:.0f}s, {nfe_fit} NFE; converged "
        f"{sum(c['converged'] for c in conv)}/{len(grp)}; n_eff {[IT.component_stats(res['means'][k], res['variances'][k])['n_effective'] for k in range(len(grp))]}; "
        f"v/v_anc {np.round(res['variances'].double().cpu().numpy().mean(1) / iv.v_anc, 3).tolist()}; "
        f"stopped {res['stopped_early']}")


# ============================================================================== fit evaluation
def _iters_of(z, meta):
    its = sorted(int(k[4:-6]) for k in z.files if k.startswith("snap") and k.endswith("_means"))
    if meta["n_iterations"] not in its:
        its.append(int(meta["n_iterations"]))
    return its


def stage_evalfits(cfg, a):
    """Stage A on fits that have teacher references: every snapshot vs teacher sets A/B (fp32)."""
    ce = cfg["evaluation"]
    root = C.out_dir(cfg)
    have_eval = lambda n, j: (root / "fits_eval" / a.split / a.variant / f"traj{n:03d}_j{j}.json").exists()
    tasks = [(n, j) for n, j in _tasks(cfg, a, a.split, have_eval)
             if C.fit_path(cfg, a.split, a.variant, n, j).with_suffix(".json").exists()
             and C.ref_path(cfg, a.split, n, j).exists()]
    log(f"evalfits {a.split}/{a.variant} shard {a.shard}/{a.n_shards}: {len(tasks)} fits")
    if not tasks:
        return
    teacher = C.build_teacher(cfg, a.device, tf32=False)
    parts = C.partition(cfg, teacher)
    bs = int(cfg["teacher"]["batch_size"])
    vid, _ = C.variant_cfg(cfg, a.variant)
    nm = int(ce["n_model"])
    for n, j in tasks:
        iv = parts[j]
        path = C.fit_path(cfg, a.split, a.variant, n, j)
        meta = json.loads(path.with_suffix(".json").read_text())
        rz = np.load(C.ref_path(cfg, a.split, n, j))
        R = int(cfg["evaluation"]["n_ref"])
        refA, refB = (rz["dino"][:R], rz["pixel"][:R]), (rz["dino"][R:], rz["pixel"][R:])
        out = dict(split=a.split, n=n, j=j, variant=a.variant, snapshots={})
        with np.load(path) as z:
            for it in _iters_of(z, meta):
                key = "means" if it == int(meta["n_iterations"]) else f"snap{it}_means"
                m = torch.as_tensor(z[key]).to(teacher.device)
                vv = torch.as_tensor(z[key.replace("means", "variances")]).to(teacher.device).float()
                g = torch.Generator(device=teacher.device).manual_seed(
                    G.seed_of(C.NS_EVAL, C.sid(a.split), n, j, vid, it))
                y, _ = IT.sample_mixture(m, vv, nm, g)
                y = y.reshape(nm, *teacher.image_shape)
                if not torch.isfinite(y).all() or meta["stopped_early"] is not None:
                    out["snapshots"][str(it)] = dict(finite=False, stopped_early=meta["stopped_early"])
                    continue
                if iv.terminal:
                    with IT.tf32(False):
                        y, _ = G.terminal_denoise(teacher, y, bs)
                f = MT.featurize(teacher, iv, y.cpu(), ce, batch_size=bs, device=a.device)
                row = MT.compare(refA, refB, f, ce, seed=n * 100 + j)
                row.update(finite=True, stats=IT.component_stats(m, vv),
                           variance_over_v_anc=float(vv.mean() / iv.v_anc))
                out["snapshots"][str(it)] = row
                log(f"evalfits {a.split}/{a.variant} traj {n} j {j} it {it}: DINOv2 excess "
                    f"{row['dinov2']['excess']:+.4f} (p {row['dinov2']['p_model']:.3f}, floor p "
                    f"{row['dinov2']['p_floor']:.3f}); pixel excess {row['pixel']['excess']:+.5f}; "
                    f"n_eff {row['stats']['n_effective']}")
        C.dump(out, root / "fits_eval" / a.split / a.variant / f"traj{n:03d}_j{j}.json")


# ============================================================================== selection
def _paired_ci(d, n_boot, seed, alpha):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def _collect(cfg, split, variant):
    root = C.out_dir(cfg) / "fits_eval" / split / variant
    rows = {}
    for p in sorted(root.glob("traj*_j*.json")):
        r = json.loads(p.read_text())
        rows[(r["n"], r["j"])] = r
    return rows


def stage_select(cfg, a):
    root = C.out_dir(cfg)
    if (root / "selection.json").exists() and not a.force:
        log("selection.json already frozen; not overwriting (use --force)")
        print((root / "selection.json").read_text())
        return
    cs = cfg["selection"]
    variants = [v["name"] for v in cfg["ibw"]["variants"]]
    data = {v: _collect(cfg, "val", v) for v in variants}
    pairs = sorted(set.intersection(*[set(d) for d in data.values()]))
    expected = int(cfg["splits"]["n_val"]) * int(cfg["partition"]["n_intervals"])
    if len(pairs) < expected and not a.force:
        raise RuntimeError(f"only {len(pairs)} of {expected} val pairs evaluated for every variant; "
                           "refusing to freeze on incomplete validation (use --force to override)")
    its = sorted({int(i) for d in data.values() for r in d.values() for i in r["snapshots"]})
    X, fails, neff = {}, {}, {}
    for v in variants:
        for it in its:
            xs, fs, ns = [], [], []
            for p in pairs:
                s = data[v][p]["snapshots"].get(str(it))
                ok = s is not None and s.get("finite", False)
                # a failed fit scores the largest attainable excess (unbiased MMD^2 <= 2 for a
                # kernel bounded by 1): ranked last, as "+inf" in the rule, but CIs stay finite
                xs.append(s["dinov2"]["excess"] if ok else 2.0)
                fs.append(not ok)
                ns.append(s["stats"]["n_effective"] if ok else 0)
            X[(v, it)], fails[(v, it)], neff[(v, it)] = np.array(xs), np.mean(fs), np.mean(ns)
    best_it = {v: min(its, key=lambda it: np.mean(X[(v, it)])) for v in variants}
    table = [dict(variant=v, iterations=it, excess_mean=float(np.mean(X[(v, it)])),
                  failure_rate=float(fails[(v, it)]), n_effective_mean=float(neff[(v, it)]))
             for v in variants for it in its]
    ref = "prior_free"
    order = sorted(variants, key=lambda v: np.mean(X[(v, best_it[v])]))
    chosen = order[0]
    rule1 = dict(best_by_mean=chosen, comparisons={})
    for v in variants:
        if v == ref:
            continue
        m, lo, hi = _paired_ci(X[(v, best_it[v])] - X[(ref, best_it[ref])], cs["n_boot"], cs["boot_seed"], cs["alpha"])
        rule1["comparisons"][f"{v} - {ref}"] = dict(mean=m, ci=[lo, hi])
    if chosen != ref:
        lo, hi = rule1["comparisons"][f"{chosen} - {ref}"]["ci"]
        if lo <= 0 <= hi:          # not separable from the reference: tie-break as pre-registered
            cands = [chosen, ref]
            chosen = min(cands, key=lambda v: (fails[(v, best_it[v])], -neff[(v, best_it[v])],
                                               np.mean(X[(v, best_it[v])])))
            rule1["tie_break"] = dict(candidates=cands, chosen=chosen,
                                      reason="CI vs prior_free contains 0: (a) failure rate, (b) n_effective")
    rule1["chosen"] = chosen
    bi = best_it[chosen]
    rule2 = dict(best=bi, comparisons={})
    pick = bi
    for it in its:
        m, lo, hi = _paired_ci(X[(chosen, it)] - X[(chosen, bi)], cs["n_boot"], cs["boot_seed"], cs["alpha"])
        rule2["comparisons"][str(it)] = dict(mean=m, ci=[lo, hi], failure_rate=float(fails[(chosen, it)]))
        if lo <= 0 and fails[(chosen, it)] <= fails[(chosen, bi)]:
            pick = it
            break
    rule2["chosen"] = int(pick)
    _, vc = C.variant_cfg(cfg, chosen)
    sel = dict(frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), variant=chosen, variant_cfg=vc,
               step_c=float(vc.get("step_c", cfg["ibw"]["step_c"])), iterations=int(pick),
               n_components=int(cfg["ibw"]["n_components"]), n_grad_samples=int(cfg["ibw"]["n_grad_samples"]),
               n_val_pairs=len(pairs), rule1=rule1, rule2=rule2, table=table,
               per_interval={str(j): {v: float(np.mean([X[(v, best_it[v])][k] for k, p in enumerate(pairs) if p[1] == j]))
                                      for v in variants} for j in sorted({p[1] for p in pairs})})
    C.dump(sel, root / "selection.json")
    log(f"selected {chosen} @ {pick} iterations (step_c {sel['step_c']})")
    print(json.dumps(dict(variant=chosen, iterations=pick, rule1=rule1, rule2=rule2), indent=2))


def stage_freeze(cfg, a):
    """Copy the selected snapshot of the selected variant's val fits to fits/val/frozen (the
    snapshot at T iterations IS the T-iteration fit: same seeds, same noise stream), adding the
    terminal pushforward bank where needed."""
    root = C.out_dir(cfg)
    sel = json.loads((root / "selection.json").read_text())
    src, it = sel["variant"], int(sel["iterations"])
    teacher = None
    ci = cfg["ibw"]
    vid, _ = C.variant_cfg(cfg, src)
    for n, _ in C.split_seeds(cfg, a.split):
        for j in range(int(cfg["partition"]["n_intervals"])):
            dst = C.fit_path(cfg, a.split, "frozen", n, j)
            srcp = C.fit_path(cfg, a.split, src, n, j)
            if dst.with_suffix(".json").exists() or not srcp.with_suffix(".json").exists():
                continue
            meta = json.loads(srcp.with_suffix(".json").read_text())
            with np.load(srcp) as z:
                key = "means" if it == int(meta["n_iterations"]) else f"snap{it}_means"
                arrays = dict(means=z[key], variances=z[key.replace("means", "variances")])
            if meta["interval"]["terminal"]:
                if teacher is None:
                    teacher = C.build_teacher(cfg, a.device, tf32=cfg["precision"]["ibw_tf32"])
                M = int(ci["terminal_bank"])
                g = torch.Generator(device=teacher.device).manual_seed(G.seed_of(C.NS_BANK, C.sid(a.split), n, j, vid))
                zz, _ = IT.sample_mixture(torch.as_tensor(arrays["means"]).to(teacher.device),
                                          torch.as_tensor(arrays["variances"]).to(teacher.device).float(), M, g)
                y, nb = G.terminal_denoise(teacher, zz.reshape(M, *teacher.image_shape), int(cfg["teacher"]["batch_size"]))
                arrays["bank"] = y.reshape(M, -1).float().cpu().numpy()
            C.save_npz_atomic(dst, **arrays)
            meta.update(variant="frozen", source_variant=src, n_iterations=it, snapshots=[],
                        component_stats=meta["snapshot_stats"].get(str(it), meta["component_stats"]),
                        frozen_from=str(srcp))
            C.dump(meta, dst.with_suffix(".json"))
    log(f"freeze {a.split}: {src} @ {it} -> fits/{a.split}/frozen")


# ============================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    ap.add_argument("stage")
    ap.add_argument("--config", default="exp5_ibw_amortized_10step_k8")
    ap.add_argument("--split", default="val")
    ap.add_argument("--variant", default="frozen")
    ap.add_argument("--trajs", type=int, nargs="*", default=None)
    ap.add_argument("--intervals", type=int, nargs="*", default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a, rest = ap.parse_known_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    stages = dict(partition=stage_partition, anchors=stage_anchors, refs=stage_refs, fit=stage_fit,
                  evalfits=stage_evalfits, select=stage_select, freeze=stage_freeze)
    if a.stage in stages:
        return stages[a.stage](cfg, a)
    from gmmd.exp5 import evaluation, report, train
    more = dict(train=train.stage_train, fitsummary=report.stage_fitsummary, **evaluation.STAGES)
    if a.stage not in more:
        raise SystemExit(f"unknown stage {a.stage!r}; known: {sorted(stages) + sorted(more)}")
    return more[a.stage](cfg, a, rest)


if __name__ == "__main__":
    main()
