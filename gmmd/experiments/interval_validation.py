"""Stage 1 of exp4: which interval does GMMD approximate best, across independent anchors?

    python -m gmmd.experiments.interval_validation --config exp4_interval_replacement \
        --shard 0 --n-shards 2          # one worker per GPU; resumable per anchor
    python -m gmmd.experiments.interval_validation --config exp4_interval_replacement --summarize

Anchors are made exactly as test images will be: ``x_T`` from a fresh seed, then the teacher
prefix ``[0, i_t)``.  At each anchor the teacher draws two independent sets of conditional
samples (A: reference, B: floor), IBW is fitted to the anchor's conditional with NO teacher
samples (prior-free init), and the fitted mixture is sampled.  Fitting and evaluation are
therefore disjoint by construction.  For the long candidate the short candidate's teacher
samples are the same rollouts snapshotted at ``i_s(short)``, which is bit-identical to stopping
there (``ScoreSDETeacher.sample_transition`` docstring).

The selection rule is fixed in the config's ``[selection]`` comment before this runs;
:func:`summarize` applies it mechanically and freezes the result in ``selection.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gmmd import diagnostics as D
from gmmd import features as Fe
from gmmd import ibw_batched
from gmmd import pipeline as P
from gmmd import provenance as Pv
from gmmd.config import load_config, parse_overrides, resolve_path
from gmmd.datasets.celebahq import data_root

__all__ = ["build_teacher", "run_shard", "summarize", "main"]

_REPO = Path(__file__).resolve().parents[2]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_teacher(cfg, device):
    from gmmd.teachers.score_sde.teacher import ScoreSDETeacher

    tc = cfg["teacher"]
    if tc["predictor"] != "reverse_diffusion" or tc["corrector"] != "none":
        raise NotImplementedError("exp4 implements the predictor-only reverse-diffusion sampler")
    return ScoreSDETeacher.from_config(tc["config"], resolve_path(tc["checkpoint"], data_root()),
                                       device=device, use_ema=bool(tc["use_ema"]),
                                       allow_tf32=bool(tc["allow_tf32"]))


def out_dir(cfg):
    return _REPO / cfg["output"]["dir"]


def denoised_features(teacher, x, s, feature, device, batch_size):
    """DINOv2 features of Tweedie-denoised states, the exp1/exp3 representation. x (n, C, H, W)."""
    den = torch.cat([teacher.tweedie_denoise(x[j:j + batch_size].to(teacher.device), s).cpu()
                     for j in range(0, len(x), batch_size)]).numpy()
    return Fe.extract(feature, den, device=device, batch_size=32), den


def anchor_metrics(fA, fB, fG, pA, pB, pG, cfg_v, seed):
    """MMD^2 (shared bandwidth from the teacher pool), permutation tests, SWD; DINOv2 + pixel."""
    out = {}
    for rep, (A, B, G) in (("dinov2", (fA, fB, fG)), ("pixel", (pA, pB, pG))):
        h = D.median_bandwidth(A, B, seed=seed)
        gA = D.mmd_permutation_test(G, A, n_perm=cfg_v["n_perm"], bandwidth=h, seed=seed)
        bA = D.mmd_permutation_test(B, A, n_perm=cfg_v["n_perm"], bandwidth=h, seed=seed)
        r = dict(bandwidth=h, mmd2_gmmd=gA["mmd2"], p_gmmd=gA["p_value"],
                 mmd2_floor=bA["mmd2"], p_floor=bA["p_value"],
                 excess=gA["mmd2"] - bA["mmd2"])
        if rep == "dinov2":
            r["swd_gmmd"] = D.sliced_wasserstein(G, A, cfg_v["n_projections"], seed=seed)["mean"]
            r["swd_floor"] = D.sliced_wasserstein(B, A, cfg_v["n_projections"], seed=seed)["mean"]
        out[rep] = r
    return out


def _intervals(cfg, teacher):
    out = {}
    for c in cfg["candidates"]:
        i_t, i_s, t, s = teacher.transition_indices(c["t"], c["s"])
        out[c["name"]] = dict(name=c["name"], i_t=i_t, i_s=i_s, t=t, s=s,
                              sigma_t=teacher.sigma(t), sigma_s=teacher.sigma(s),
                              n_steps=i_s - i_t, n_anchors=int(c["n_anchors"]))
    starts = {v["i_t"] for v in out.values()}
    if len(starts) != 1:
        raise NotImplementedError("the shared-rollout design needs every candidate to start at one i_t")
    return out


def run_shard(cfg, shard, n_shards, device):
    teacher = build_teacher(cfg, device)
    iv = _intervals(cfg, teacher)
    cv, cg = cfg["validation"], cfg["gmmd"]
    bs = int(cfg["teacher"]["batch_size"])
    root = out_dir(cfg) / "validation"
    (root / "anchors").mkdir(parents=True, exist_ok=True)
    n_anchor = max(v["n_anchors"] for v in iv.values())
    mine = [a for a in range(n_anchor) if a % n_shards == shard
            and not (root / "anchors" / f"anchor{a:03d}.json").exists()]
    log(f"shard {shard}/{n_shards} on {device}: anchors {mine}")
    if not mine:
        return
    i_t = next(iter(iv.values()))["i_t"]
    ends = sorted({v["i_s"] for v in iv.values()})
    n = int(cv["n_per_set"])

    # ---- anchors: x_T and the teacher prefix, batched ------------------------------------------
    seeds = [cv["xT_seed_base"] + a for a in mine]
    with P.Timer(device) as tm:
        x, _, nfe_prefix = P.fine_steps(teacher, P.prior_sample(teacher, seeds), 0, i_t,
                                        P.Generators(seeds, "prefix", device), batch_size=bs)
    x_t = x.detach()
    log(f"prefix [0, {i_t}) for {len(mine)} anchors: {tm.seconds:.0f}s")
    prefix_s_per_anchor = tm.seconds / len(mine)

    # ---- per anchor: 2n teacher rollouts to the last end it needs, snapshotting the others -----
    teach = {}           # (a, name) -> (n_rollouts, C, H, W) cpu
    teach_cost = {}
    for k, a in enumerate(mine):
        need = [v for v in iv.values() if a < v["n_anchors"]]
        last = max(v["i_s"] for v in need)
        rseeds = [cv["rollout_seed_base"] + 1000 * a + r for r in range(2 * n)]
        gens = P.Generators(rseeds, "segment", device)
        xr = x_t[k:k + 1].expand(2 * n, *teacher.image_shape).contiguous()
        cur = i_t
        for end in [e for e in ends if e <= last]:
            with P.Timer(device) as tm:
                xr, _, nfe = P.fine_steps(teacher, xr, cur, end, gens, batch_size=bs)
            for v in need:
                if v["i_s"] == end:
                    teach[(a, v["name"])] = xr.detach().cpu()
                    teach_cost[(a, v["name"])] = dict(seconds=tm.seconds, nfe=nfe)
            cur = end
        log(f"anchor {a}: teacher rollouts to {last} done")

    # ---- per candidate: batched IBW over this shard's anchors, then evaluate snapshots ----------
    results = {a: dict(anchor=a, xT_seed=cv["xT_seed_base"] + a, candidates={}) for a in mine}
    for name, v in iv.items():
        idx = [k for k, a in enumerate(mine) if a < v["n_anchors"]]
        if not idx:
            continue
        anchors = [mine[k] for k in idx]
        target = ibw_batched.VEConditionalScores(teacher, x_t[idx], v["t"], v["s"], batch_size=bs)
        K = int(cg["n_components"])
        init_m, init_v = [], []
        for a, k in zip(anchors, idx):
            rng = np.random.default_rng(cv["gmmd_seed_base"] + a)
            z = rng.standard_normal((K, target.dim))
            init_m.append(x_t[k].reshape(-1).double().cpu().numpy()[None]
                          + float(cg["init_spread"]) * np.sqrt(target.gap) * z)
            init_v.append(np.full(K, target.gap))
        with P.Timer(device) as tm:
            fit = ibw_batched.fit_batched(target, np.stack(init_m), np.stack(init_v),
                                          [cv["gmmd_seed_base"] + a for a in anchors],
                                          n_iterations=int(cg["n_iterations"]),
                                          step_size=float(cg["step_size"]),
                                          n_grad_samples=int(cg["n_grad_samples"]),
                                          snapshot_at=cg["snapshot_iterations"], device=device)
        log(f"{name}: batched IBW over {len(anchors)} anchors, {tm.seconds:.0f}s, "
            f"{target.nfe} score examples, stopped {fit['stopped_early']}")
        fit_s_per_anchor = tm.seconds / len(anchors)
        nfe_per_anchor = target.nfe // len(anchors)

        for j, (a, k) in enumerate(zip(anchors, idx)):
            T_all = teach[(a, name)]
            TA, TB = T_all[:n], T_all[n:]
            fA, _ = denoised_features(teacher, TA, v["s"], cv["feature"], device, bs)
            fB, _ = denoised_features(teacher, TB, v["s"], cv["feature"], device, bs)
            pA = Fe.extract("pixel", TA.numpy(), downsample=cv["pixel_downsample"])
            pB = Fe.extract("pixel", TB.numpy(), downsample=cv["pixel_downsample"])
            rec = dict(interval=v, teacher_cost=teach_cost[(a, name)],
                       prefix_seconds_per_anchor=prefix_s_per_anchor, nfe_prefix_per_anchor=i_t,
                       fit_seconds_per_anchor=fit_s_per_anchor, fit_nfe_per_anchor=nfe_per_anchor,
                       stopped_early=fit["stopped_early"][j],
                       grad_norm_first=float(fit["grad_norms"][0, j]),
                       grad_norm_last=float(fit["grad_norms"][-1, j]),
                       floor_teacher=None, snapshots={})
            for it, (m, var) in sorted(fit["snapshots"].items()):
                g = torch.Generator(device=device).manual_seed(cv["gmmd_seed_base"] + 10000 + a)
                xg = ibw_batched.sample_batched(m[j:j + 1], var[j:j + 1], n, [g])[0]
                xg = xg.reshape(n, *teacher.image_shape)
                finite = bool(torch.isfinite(xg).all())
                if not finite:
                    rec["snapshots"][str(it)] = dict(finite=False)
                    continue
                fG, _ = denoised_features(teacher, xg, v["s"], cv["feature"], device, bs)
                pG = Fe.extract("pixel", xg.cpu().numpy(), downsample=cv["pixel_downsample"])
                m_ = anchor_metrics(fA, fB, fG, pA, pB, pG, cv, seed=a)
                m_.update(finite=True, variance=float(var[j].mean()))
                rec["snapshots"][str(it)] = m_
                log(f"anchor {a} {name} it={it}: DINOv2 MMD^2 gmmd {m_['dinov2']['mmd2_gmmd']:+.5f} "
                    f"floor {m_['dinov2']['mmd2_floor']:+.5f} (p {m_['dinov2']['p_gmmd']:.3f} / "
                    f"{m_['dinov2']['p_floor']:.3f}); SWD {m_['dinov2']['swd_gmmd']:.3f} / "
                    f"{m_['dinov2']['swd_floor']:.3f}")
            results[a]["candidates"][name] = rec

    for a, r in results.items():
        r["provenance"] = Pv.snapshot(cfg, device)
        Pv.dump_json(r, root / "anchors" / f"anchor{a:03d}.json")
        k = mine.index(a)
        np.save(root / "anchors" / f"anchor{a:03d}_xt.npy", x_t[k].cpu().numpy())
    log("shard done")


# ============================================================================== summarize
def _paired_ci(d, n_boot, seed, alpha):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def summarize(cfg, force=False):
    root = out_dir(cfg) / "validation"
    if (root / "selection.json").exists() and not force:
        log(f"{root / 'selection.json'} already frozen; not overwriting (use --force-summary)")
        return json.loads((root / "selection.json").read_text()), None
    recs = [json.loads(p.read_text()) for p in sorted((root / "anchors").glob("anchor*.json"))]
    for c in cfg["candidates"]:
        have = sum(c["name"] in r["candidates"] for r in recs)
        if have < int(c["n_anchors"]):
            raise RuntimeError(f"candidate {c['name']}: {have} of {c['n_anchors']} anchors done; "
                               "refusing to freeze a selection on incomplete validation")
    cs, cg = cfg["selection"], cfg["gmmd"]
    its = [str(i) for i in cg["snapshot_iterations"]]
    final_it = str(cg["n_iterations"])
    rows, per = [], {}
    for name in [c["name"] for c in cfg["candidates"]]:
        rs = [r for r in recs if name in r["candidates"]]
        for it in its:
            vals = []
            for r in rs:
                c = r["candidates"][name]
                s = c["snapshots"].get(it, {})
                ok = s.get("finite", False) and c["stopped_early"] is None
                vals.append(dict(anchor=r["anchor"], ok=ok,
                                 X=s["dinov2"]["excess"] if ok else np.nan,
                                 E=s["dinov2"]["mmd2_gmmd"] if ok else np.nan,
                                 F=s["dinov2"]["mmd2_floor"] if ok else np.nan,
                                 det=s["dinov2"]["p_gmmd"] < cs["alpha"] if ok else True,
                                 det_floor=s["dinov2"]["p_floor"] < cs["alpha"] if ok else False,
                                 swd=s["dinov2"]["swd_gmmd"] if ok else np.nan,
                                 swd_floor=s["dinov2"]["swd_floor"] if ok else np.nan,
                                 pixX=s["pixel"]["excess"] if ok else np.nan,
                                 pix_det=s["pixel"]["p_gmmd"] < cs["alpha"] if ok else True))
            per[(name, it)] = vals
            c0 = rs[0]["candidates"][name]
            iv = c0["interval"]
            arr = lambda k: np.array([v[k] for v in vals], float)
            fail = 1 - np.mean([v["ok"] for v in vals])
            rows.append(dict(
                candidate=name, iterations=int(it), i_t=iv["i_t"], i_s=iv["i_s"],
                t=iv["t"], s=iv["s"], sigma_t=iv["sigma_t"], sigma_s=iv["sigma_s"],
                teacher_steps_replaced=iv["n_steps"], n_anchors=len(vals),
                mmd2_gmmd_mean=np.nanmean(arr("E")), mmd2_gmmd_sd=np.nanstd(arr("E"), ddof=1),
                mmd2_floor_mean=np.nanmean(arr("F")), mmd2_floor_sd=np.nanstd(arr("F"), ddof=1),
                excess_mean=np.nanmean(arr("X")), excess_sd=np.nanstd(arr("X"), ddof=1),
                swd_gmmd_mean=np.nanmean(arr("swd")), swd_floor_mean=np.nanmean(arr("swd_floor")),
                pixel_excess_mean=np.nanmean(arr("pixX")),
                detection_rate=np.mean(arr("det")), floor_detection_rate=np.mean(arr("det_floor")),
                pixel_detection_rate=np.mean(arr("pix_det")),
                numerical_failure_rate=fail,
                gmmd_nfe_per_anchor=int(c0["fit_nfe_per_anchor"] * int(it) / int(final_it)),
                teacher_segment_nfe_per_sample=iv["n_steps"],
                fit_seconds_per_anchor_at_300=float(np.mean([r["candidates"][name]["fit_seconds_per_anchor"] for r in rs])),
            ))

    # ---- rule 1: interval, on shared anchors, at the final iteration count ---------------------
    names = [c["name"] for c in cfg["candidates"]]
    shared = sorted(set.intersection(*[{v["anchor"] for v in per[(nm, final_it)]} for nm in names]))
    xs = {nm: {v["anchor"]: v for v in per[(nm, final_it)]} for nm in names}
    assert len(names) == 2
    a_, b_ = names
    d = [xs[a_][k]["X"] - xs[b_][k]["X"] for k in shared]
    mean, lo, hi = _paired_ci(d, cs["n_boot"], cs["boot_seed"], cs["alpha"])
    rule1 = dict(shared_anchors=shared, diff=f"X[{a_}] - X[{b_}]", mean=mean, ci=[lo, hi])
    if hi < 0:
        chosen, why = a_, f"{a_} has lower excess MMD^2 on all shared anchors' mean (CI excludes 0)"
    elif lo > 0:
        chosen, why = b_, f"{b_} has lower excess MMD^2 (CI excludes 0)"
    else:
        def key(nm):
            r = next(r for r in rows if r["candidate"] == nm and str(r["iterations"]) == final_it)
            return (r["numerical_failure_rate"], r["detection_rate"], -r["teacher_steps_replaced"])
        chosen = min(names, key=key)
        why = "CI contains 0; tie broken by failure rate, detection rate, then longer interval"
    rule1.update(chosen=chosen, reason=why)

    # ---- rule 2: iterations within the chosen interval -----------------------------------------
    sub = {it: per[(chosen, it)] for it in its}
    best_it = min(its, key=lambda it: np.nanmean([v["X"] for v in sub[it]]))
    fail = {it: 1 - np.mean([v["ok"] for v in sub[it]]) for it in its}
    rule2 = dict(best=best_it, comparisons={})
    pick = best_it
    for it in sorted(its, key=int):
        dd = [u["X"] - w["X"] for u, w in zip(sub[it], sub[best_it])]
        m_, lo_, hi_ = _paired_ci(dd, cs["n_boot"], cs["boot_seed"], cs["alpha"])
        rule2["comparisons"][it] = dict(mean=m_, ci=[lo_, hi_], failure_rate=fail[it])
        if lo_ <= 0 and fail[it] <= fail[best_it] and int(it) <= int(pick):
            pick = it
            break
    rule2["chosen_iterations"] = int(pick)

    iv = next(r for r in rows if r["candidate"] == chosen and str(r["iterations"]) == pick)
    selection = dict(
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        interval=dict(name=chosen, t=iv["t"], s=iv["s"], i_t=iv["i_t"], i_s=iv["i_s"],
                      sigma_t=iv["sigma_t"], sigma_s=iv["sigma_s"],
                      teacher_steps=iv["teacher_steps_replaced"]),
        gmmd=dict(n_components=cg["n_components"], n_grad_samples=cg["n_grad_samples"],
                  step_size=cg["step_size"], n_iterations=int(pick), init=cg["init"],
                  init_spread=cg["init_spread"]),
        rule1=rule1, rule2=rule2)
    Pv.dump_json(selection, root / "selection.json")
    Pv.dump_json(rows, root / "candidates.json")
    _write_table(rows, root / "candidates.md")
    log(f"selected: {chosen}, {pick} iterations")
    return selection, rows


def _write_table(rows, path):
    hdr = ("| candidate | grid i_t→i_s | t → s | σ_t → σ_s | steps replaced | iters | anchors | "
           "GMMD MMD² mean ± SD | teacher floor MMD² mean ± SD | excess mean ± SD | SWD GMMD / floor | "
           "detect. rate GMMD / floor | pixel detect. | num. fail | GMMD NFE/anchor | fit s/anchor (300 it) |")
    lines = [hdr, "|" + "---|" * 16]
    for r in rows:
        lines.append(
            f"| {r['candidate']} | {r['i_t']}→{r['i_s']} | {r['t']:.4f} → {r['s']:.4f} | "
            f"{r['sigma_t']:.3f} → {r['sigma_s']:.3f} | {r['teacher_steps_replaced']} | {r['iterations']} | "
            f"{r['n_anchors']} | {r['mmd2_gmmd_mean']:.4f} ± {r['mmd2_gmmd_sd']:.4f} | "
            f"{r['mmd2_floor_mean']:.4f} ± {r['mmd2_floor_sd']:.4f} | "
            f"{r['excess_mean']:.4f} ± {r['excess_sd']:.4f} | "
            f"{r['swd_gmmd_mean']:.3f} / {r['swd_floor_mean']:.3f} | "
            f"{r['detection_rate']:.2f} / {r['floor_detection_rate']:.2f} | {r['pixel_detection_rate']:.2f} | "
            f"{r['numerical_failure_rate']:.2f} | {r['gmmd_nfe_per_anchor']} | "
            f"{r['fit_seconds_per_anchor_at_300']:.1f} |")
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="exp4_interval_replacement")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--force-summary", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    if a.summarize:
        summarize(cfg, force=a.force_summary)
    else:
        run_shard(cfg, a.shard, a.n_shards, a.device)


if __name__ == "__main__":
    main()
