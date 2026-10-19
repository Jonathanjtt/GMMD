"""Stage 2 of exp4: replace the frozen interval inside full image generation and compare methods.

    # every stage is resumable per 64-image chunk and shards chunks over workers (one per GPU)
    python -m gmmd.experiments.interval_replacement prefix  --n-chunks 3 --shard 0 --n-shards 2
    python -m gmmd.experiments.interval_replacement method  --method teacher --n-chunks 3 ...
    python -m gmmd.experiments.interval_replacement method  --method coarse  ...
    python -m gmmd.experiments.interval_replacement method  --method gmmd    ...
    python -m gmmd.experiments.interval_replacement e2e     --method gmmd     # uncached timing
    python -m gmmd.experiments.interval_replacement evaluate --n-chunks 3 --tag pilot

Methods (all share x_T and the cached prefix state x_t of each test image, and the suffix noise
streams; each draws its own segment noise):

  teacher  original: fine reverse-diffusion steps [i_t, i_s), then the suffix
  coarse   ``coarse_steps`` SMLD steps over [i_t, i_s), then the same suffix
  gmmd     one IBW fit per image to p(x_s | x_t) (frozen config from selection.json), ONE random
           draw from that image's own mixture, then the same suffix

Sharing suffix noise is common random numbers: each method's suffix is still the teacher's
transition kernel from its own x_s, so each method's marginal law is unchanged.

Cost accounting: every stage is timed with the device synchronised, at the same network batch
(``teacher.batch_size`` images per call), and its score-network examples are counted.  The
prefix is computed once per image and cached; it is added back to every method's full cost.
``e2e`` additionally times x_T -> x_0 without any cache.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gmmd import fid as FID
from gmmd import ibw_batched
from gmmd import pipeline as P
from gmmd import provenance as Pv
from gmmd.config import load_config, parse_overrides
from gmmd.datasets.celebahq import data_root
from gmmd.experiments.interval_validation import build_teacher, log, out_dir

__all__ = ["main"]

METHODS = ("teacher", "coarse", "gmmd")


def _selection(cfg):
    p = out_dir(cfg) / "validation" / "selection.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run interval_validation --summarize first "
                                "(the interval and GMMD config are frozen there)")
    return json.loads(p.read_text())


def _chunk_seeds(cfg, c):
    g = cfg["generation"]
    return [int(g["xT_seed_base"]) + c * int(g["chunk"]) + k for k in range(int(g["chunk"]))]


def _mine(n_chunks, shard, n_shards, done):
    return [c for c in range(n_chunks) if c % n_shards == shard and not done(c)]


# ============================================================================== segments
def run_segment(method, teacher, x_t, seeds, sel, cfg, device):
    """x_t (B, C, H, W) -> (x_s, record).  Record holds seconds, NFE, peak memory per sub-stage."""
    bs = int(cfg["teacher"]["batch_size"])
    i_t, i_s = sel["interval"]["i_t"], sel["interval"]["i_s"]
    rec = {}
    if method == "teacher":
        with P.Timer(device) as tm:
            x_s, _, nfe = P.fine_steps(teacher, x_t, i_t, i_s, P.Generators(seeds, "segment", device), bs)
        rec["segment"] = dict(seconds=tm.seconds, nfe=nfe, peak_gb=tm.peak_gb, steps=i_s - i_t)
    elif method == "coarse":
        n = int(cfg["generation"]["coarse_steps"])
        with P.Timer(device) as tm:
            x_s, _, nfe = P.coarse_steps(teacher, x_t, i_t, i_s, n, P.Generators(seeds, "coarse", device), bs)
        rec["segment"] = dict(seconds=tm.seconds, nfe=nfe, peak_gb=tm.peak_gb, steps=n,
                              blocks=P.coarse_blocks(i_t, i_s, n))
    elif method == "gmmd":
        g = sel["gmmd"]
        K = int(g["n_components"])
        with P.Timer(device) as tm_fit:
            target = ibw_batched.VEConditionalScores(teacher, x_t, sel["interval"]["t"],
                                                     sel["interval"]["s"], batch_size=bs)
            fseeds = [P.stage_seed(s, "gmmd_fit") for s in seeds]
            xt_flat = x_t.reshape(len(seeds), -1)
            # prior-free init N(x_t, gap I): built on device from a per-image generator
            gens = [torch.Generator(device=device).manual_seed(fs) for fs in fseeds]
            z = torch.stack([torch.randn((K, target.dim), generator=gg, device=device) for gg in gens])
            init_m = xt_flat[:, None, :] + float(g["init_spread"]) * float(np.sqrt(target.gap)) * z
            init_v = torch.full((len(seeds), K), float(target.gap), device=device)
            fit = ibw_batched.fit_batched(target, init_m, init_v, [fs + 1 for fs in fseeds],
                                          n_iterations=int(g["n_iterations"]),
                                          step_size=float(g["step_size"]),
                                          n_grad_samples=int(g["n_grad_samples"]), device=device)
        with P.Timer(device) as tm_s:
            sg = [torch.Generator(device=device).manual_seed(P.stage_seed(s, "gmmd_sample")) for s in seeds]
            x_s = ibw_batched.sample_batched(fit["means"], fit["variances"], 1, sg)[:, 0]
            x_s = x_s.reshape(len(seeds), *teacher.image_shape)
        n_fail = sum(v is not None for v in fit["stopped_early"])
        rec["gmmd_fit"] = dict(seconds=tm_fit.seconds, nfe=target.nfe, peak_gb=tm_fit.peak_gb,
                               iterations=int(g["n_iterations"]), stopped_early=fit["stopped_early"],
                               n_failed=n_fail, grad_norm_first=fit["grad_norms"][0].tolist(),
                               grad_norm_last=fit["grad_norms"][-1].tolist(),
                               variance_final=fit["variances"][:, 0].tolist())
        rec["gmmd_sample"] = dict(seconds=tm_s.seconds, nfe=0, peak_gb=tm_s.peak_gb)
        rec["segment"] = dict(seconds=tm_fit.seconds + tm_s.seconds, nfe=target.nfe,
                              peak_gb=max(tm_fit.peak_gb, tm_s.peak_gb))
    else:
        raise ValueError(method)
    if not torch.isfinite(x_s).all():
        rec["non_finite_x_s"] = int((~torch.isfinite(x_s)).reshape(len(seeds), -1).any(1).sum())
    return x_s, rec


def run_suffix(teacher, x_s, seeds, sel, device, bs):
    with P.Timer(device) as tm:
        _, x0, nfe = P.fine_steps(teacher, x_s, sel["interval"]["i_s"], teacher.sde.N,
                                  P.Generators(seeds, "suffix", device), bs)
    return x0, dict(seconds=tm.seconds, nfe=nfe, peak_gb=tm.peak_gb,
                    steps=teacher.sde.N - sel["interval"]["i_s"])


def run_prefix(teacher, seeds, i_t, device, bs):
    with P.Timer(device) as tm:
        x_t, _, nfe = P.fine_steps(teacher, P.prior_sample(teacher, seeds), 0, i_t,
                                   P.Generators(seeds, "prefix", device), bs)
    return x_t, dict(seconds=tm.seconds, nfe=nfe, peak_gb=tm.peak_gb, steps=i_t)


def _range_stats(x0):
    x = x0.detach()
    return dict(min=float(x.min()), max=float(x.max()), mean=float(x.mean()),
                frac_below0=float((x < 0).float().mean()), frac_above1=float((x > 1).float().mean()),
                finite=bool(torch.isfinite(x).all()), shape=list(x.shape))


# ============================================================================== stages
def stage_prefix(cfg, a, device):
    sel = _selection(cfg)
    teacher = build_teacher(cfg, device)
    root = out_dir(cfg) / "generation" / "prefix"
    root.mkdir(parents=True, exist_ok=True)
    bs = int(cfg["teacher"]["batch_size"])
    for c in _mine(a.n_chunks, a.shard, a.n_shards, lambda c: (root / f"chunk{c:03d}.npz").exists()):
        seeds = _chunk_seeds(cfg, c)
        x_t, rec = run_prefix(teacher, seeds, sel["interval"]["i_t"], device, bs)
        np.savez(root / f"chunk{c:03d}.tmp.npz", x_t=x_t.cpu().numpy(), seeds=np.asarray(seeds))
        (root / f"chunk{c:03d}.tmp.npz").rename(root / f"chunk{c:03d}.npz")
        rec.update(chunk=c, n=len(seeds), gpu=torch.cuda.get_device_name(device))
        Pv.dump_json(rec, root / f"chunk{c:03d}.json")
        log(f"prefix chunk {c}: {rec['seconds']:.0f}s, {rec['nfe']} NFE, peak {rec['peak_gb']:.1f} GB")


def stage_method(cfg, a, device):
    sel = _selection(cfg)
    teacher = build_teacher(cfg, device)
    gen = out_dir(cfg) / "generation"
    root = gen / a.method
    root.mkdir(parents=True, exist_ok=True)
    bs = int(cfg["teacher"]["batch_size"])
    for c in _mine(a.n_chunks, a.shard, a.n_shards, lambda c: (root / f"chunk{c:03d}.npz").exists()):
        pz = gen / "prefix" / f"chunk{c:03d}.npz"
        if not pz.exists():
            log(f"chunk {c}: prefix missing, skipped")
            continue
        z = np.load(pz)
        seeds = [int(s) for s in z["seeds"]]
        x_t = torch.as_tensor(z["x_t"], device=device)
        x_s, rec = run_segment(a.method, teacher, x_t, seeds, sel, cfg, device)
        x0, rec["suffix"] = run_suffix(teacher, x_s, seeds, sel, device, bs)
        rec.update(chunk=c, n=len(seeds), method=a.method, range=_range_stats(x0),
                   gpu=torch.cuda.get_device_name(device),
                   prefix=json.loads((gen / "prefix" / f"chunk{c:03d}.json").read_text()))
        np.savez(root / f"chunk{c:03d}.tmp.npz", images=FID.to_uint8_upstream(x0),
                 x_s=x_s.cpu().numpy(), seeds=np.asarray(seeds))
        (root / f"chunk{c:03d}.tmp.npz").rename(root / f"chunk{c:03d}.npz")
        Pv.dump_json(rec, root / f"chunk{c:03d}.json")
        log(f"{a.method} chunk {c}: segment {rec['segment']['seconds']:.0f}s / {rec['segment']['nfe']} NFE, "
            f"suffix {rec['suffix']['seconds']:.0f}s; x0 range [{rec['range']['min']:.3f}, "
            f"{rec['range']['max']:.3f}]")


def stage_e2e(cfg, a, device):
    """Uncached x_T -> x_0 for one chunk of fresh seeds, one method per call: the real full cost."""
    sel = _selection(cfg)
    teacher = build_teacher(cfg, device)
    bs = int(cfg["teacher"]["batch_size"])
    root = out_dir(cfg) / "generation" / "e2e"
    root.mkdir(parents=True, exist_ok=True)
    seeds = [2_000_000 + k for k in range(int(cfg["generation"]["chunk"]))]
    with P.Timer(device) as tm:
        x_t, r_pre = run_prefix(teacher, seeds, sel["interval"]["i_t"], device, bs)
        x_s, r_seg = run_segment(a.method, teacher, x_t, seeds, sel, cfg, device)
        x0, r_suf = run_suffix(teacher, x_s, seeds, sel, device, bs)
    rec = dict(method=a.method, n=len(seeds), seconds=tm.seconds, peak_gb=tm.peak_gb,
               nfe=r_pre["nfe"] + r_seg["segment"]["nfe"] + r_suf["nfe"],
               prefix=r_pre, suffix=r_suf, range=_range_stats(x0), **r_seg,
               gpu=torch.cuda.get_device_name(device), provenance=Pv.snapshot(cfg, device))
    Pv.dump_json(rec, root / f"{a.method}.json")
    log(f"e2e {a.method}: {tm.seconds:.0f}s for {len(seeds)} images ({tm.seconds / len(seeds):.1f} s/img), "
        f"{rec['nfe'] / len(seeds):.0f} NFE/img, peak {tm.peak_gb:.1f} GB")


# ============================================================================== evaluation
def reference_features(cfg, device):
    """Inception (FID) and VGG16 (P/R) features of all reference images, cached under data/."""
    cache = data_root() / "fid_cache"
    cache.mkdir(parents=True, exist_ok=True)
    splits = cfg["fid"]["reference_splits"]
    tag = "celebahq256_" + "+".join(splits)
    paths = {k: cache / f"{tag}_{k}.npy" for k in ("fid", "prc")}
    if not all(p.exists() for p in paths.values()):
        feats = {"fid": [], "prc": []}
        n = 0
        for batch in FID.iter_reference(data_root() / "celebahq256", splits):
            for k in feats:
                feats[k].append(FID.extract(batch, k, device))
            n += len(batch)
            if n % 2560 == 0:
                log(f"reference features: {n} images")
        for k, p in paths.items():
            np.save(p, np.concatenate(feats[k]))
    return {k: np.load(p) for k, p in paths.items()}


def stage_evaluate(cfg, a, device):
    gen = out_dir(cfg) / "generation"
    chunks = [c for c in range(a.n_chunks)
              if all((gen / m / f"chunk{c:03d}.npz").exists() for m in METHODS)]
    if len(chunks) != a.n_chunks:
        log(f"evaluating {len(chunks)} of {a.n_chunks} chunks complete for every method")
    ref = reference_features(cfg, device)
    from torch_fidelity.metric_fid import fid_features_to_statistics
    ref_stats = fid_features_to_statistics(torch.as_tensor(ref["fid"].astype(np.float64)))
    imgs = {m: np.concatenate([np.load(gen / m / f"chunk{c:03d}.npz")["images"] for c in chunks])
            for m in METHODS}
    n = len(imgs["teacher"])
    rng = np.random.default_rng(int(cfg["fid"]["pr_subset_seed"]))
    real_idx = rng.choice(len(ref["prc"]), n, replace=False)
    out = dict(tag=a.tag, n_images=n, chunks=chunks, n_reference=int(len(ref["fid"])),
               reference_splits=cfg["fid"]["reference_splits"], pr_real_subset_size=n,
               fid_extractor="inception-v3-compat/2048", pr_extractor="vgg16/fc2_relu",
               pr_k=int(cfg["fid"]["pr_k"]), methods={})
    feats = {}
    for m in METHODS:
        f_fid = FID.extract(imgs[m], "fid", device)
        f_prc = FID.extract(imgs[m], "prc", device)
        feats[m] = f_fid
        out["methods"][m] = dict(fid=FID.fid(f_fid, ref_stats),
                                 **FID.precision_recall(f_prc, ref["prc"][real_idx],
                                                        k=int(cfg["fid"]["pr_k"]), device=device))
        log(f"{a.tag} {m}: FID {out['methods'][m]['fid']:.2f}  P {out['methods'][m]['precision']:.3f}  "
            f"R {out['methods'][m]['recall']:.3f}  (n = {n})")
    boot = FID.bootstrap_fid(feats, ref_stats, int(cfg["fid"]["n_boot"]), 0)
    for m in METHODS:
        out["methods"][m]["bootstrap"] = boot[m]
    # real-vs-real control at the same n: FID of n random reference images vs the full reference
    f_real_n = ref["fid"][rng.choice(len(ref["fid"]), n, replace=False)]
    out["real_subset_fid_same_n"] = FID.fid(f_real_n, ref_stats)
    out["costs"] = cost_summary(cfg, chunks)
    out["provenance"] = Pv.snapshot(cfg, device)
    Pv.dump_json(out, out_dir(cfg) / f"metrics_{a.tag}.json")
    _grids(gen, a.tag, out_dir(cfg) / "figures")


def cost_summary(cfg, chunks):
    gen = out_dir(cfg) / "generation"
    out = {}
    pre = [json.loads((gen / "prefix" / f"chunk{c:03d}.json").read_text()) for c in chunks]
    n = sum(p["n"] for p in pre)
    pre_s = sum(p["seconds"] for p in pre)
    pre_nfe = sum(p["nfe"] for p in pre)
    for m in METHODS:
        rs = [json.loads((gen / m / f"chunk{c:03d}.json").read_text()) for c in chunks]
        seg_s = sum(r["segment"]["seconds"] for r in rs)
        suf_s = sum(r["suffix"]["seconds"] for r in rs)
        seg_nfe = sum(r["segment"]["nfe"] for r in rs)
        suf_nfe = sum(r["suffix"]["nfe"] for r in rs)
        d = dict(n_images=n,
                 segment_s_per_img=seg_s / n, segment_nfe_per_img=seg_nfe / n,
                 prefix_s_per_img=pre_s / n, prefix_nfe_per_img=pre_nfe / n,
                 suffix_s_per_img=suf_s / n, suffix_nfe_per_img=suf_nfe / n,
                 full_s_per_img=(pre_s + seg_s + suf_s) / n,
                 full_nfe_per_img=(pre_nfe + seg_nfe + suf_nfe) / n,
                 throughput_img_per_gpu_hour=3600 * n / (pre_s + seg_s + suf_s),
                 peak_gb=max(max(r["segment"]["peak_gb"], r["suffix"]["peak_gb"]) for r in rs + []),
                 non_finite=sum(r.get("non_finite_x_s", 0) for r in rs),
                 x0_range=[min(r["range"]["min"] for r in rs), max(r["range"]["max"] for r in rs)],
                 frac_clipped=float(np.mean([r["range"]["frac_below0"] + r["range"]["frac_above1"] for r in rs])))
        if m == "gmmd":
            d["fit_s_per_img"] = sum(r["gmmd_fit"]["seconds"] for r in rs) / n
            d["sample_s_per_img"] = sum(r["gmmd_sample"]["seconds"] for r in rs) / n
            d["fit_failures"] = sum(r["gmmd_fit"]["n_failed"] for r in rs)
        d["peak_gb"] = max(d["peak_gb"], max(p["peak_gb"] for p in pre))
        e2e = gen / "e2e" / f"{m}.json"
        if e2e.exists():
            e = json.loads(e2e.read_text())
            d["e2e_uncached_s_per_img"] = e["seconds"] / e["n"]
            d["e2e_peak_gb"] = e["peak_gb"]
        out[m] = d
    return out


def _grids(gen, tag, fig_dir):
    from PIL import Image

    fig_dir.mkdir(parents=True, exist_ok=True)
    first = {m: np.load(gen / m / "chunk000.npz")["images"] for m in METHODS}
    for m, im in first.items():                       # first 64 images, no selection
        g = im[:64].reshape(8, 8, 256, 256, 3).transpose(0, 2, 1, 3, 4).reshape(8 * 256, 8 * 256, 3)
        Image.fromarray(g).resize((1024, 1024), Image.BILINEAR).save(fig_dir / f"grid_{m}_{tag}.png")
    # same x_T, three methods side by side: rows = images 0..11, columns = methods
    rows = np.concatenate([np.concatenate([first[m][i] for m in METHODS], axis=1) for i in range(12)], axis=0)
    Image.fromarray(rows).save(fig_dir / f"paired_{tag}.png")


def stage_table(cfg, a, device):
    """The report's results table, rendered from metrics_<tag>.json (no hand-copied numbers)."""
    mt = json.loads((out_dir(cfg) / f"metrics_{a.tag}.json").read_text())
    sel = _selection(cfg)
    iv, g = sel["interval"], sel["gmmd"]
    span = f"grid {iv['i_t']}→{iv['i_s']} (t {iv['t']:.4f}→{iv['s']:.4f})"
    desc = dict(teacher=f"{iv['teacher_steps']} fine SMLD steps (original)",
                coarse=f"{cfg['generation']['coarse_steps']} coarse SMLD steps",
                gmmd=f"IBW K={g['n_components']}, B={g['n_grad_samples']}, {g['n_iterations']} it × "
                     f"{g['step_size']}, prior-free init, 1 draw")
    lines = ["| Method | Replaced interval | Segment steps / GMMD configuration | Number of images | FID ↓ "
             "| Precision ↑ | Recall ↑ | Segment cost ↓ | Full generation cost ↓ | Peak memory |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        r, c = mt["methods"][m], mt["costs"][m]
        full = f"{c['full_s_per_img']:.1f} s/img, {c['full_nfe_per_img']:.0f} NFE/img"
        if "e2e_uncached_s_per_img" in c:
            full += f" (uncached e2e {c['e2e_uncached_s_per_img']:.1f} s/img)"
        lines.append(
            f"| {dict(teacher='A. Original teacher', coarse='B. Coarse teacher', gmmd='C. GMMD replacement')[m]} "
            f"| {'none (original)' if m == 'teacher' else span} | {desc[m]} | {mt['n_images']} "
            f"| {r['fid']:.2f} ± {r['bootstrap']['sd']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} "
            f"| {c['segment_s_per_img']:.2f} s/img, {c['segment_nfe_per_img']:.0f} NFE/img | {full} "
            f"| {c['peak_gb']:.1f} GB |")
    path = out_dir(cfg) / f"results_table_{a.tag}.md"
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text())


def stage_check(cfg, a, device):
    """Gate between the pilot and the full run: pipeline integrity and numerical stability only.

    Passing says the pipeline ran correctly, NOT that GMMD is good.  Exits non-zero on failure.
    """
    import sys

    gen = out_dir(cfg) / "generation"
    problems = []
    for m in ("prefix",) + METHODS:
        for c in range(a.n_chunks):
            j = gen / m / f"chunk{c:03d}.json"
            if not j.exists() or not (gen / m / f"chunk{c:03d}.npz").exists():
                problems.append(f"{m} chunk {c} missing")
                continue
            if m == "prefix":
                continue
            r = json.loads(j.read_text())
            im = np.load(gen / m / f"chunk{c:03d}.npz")["images"]
            if im.dtype != np.uint8 or im.shape[1:] != (256, 256, 3) or len(im) != r["n"]:
                problems.append(f"{m} chunk {c}: images {im.dtype} {im.shape}")
            if not r["range"]["finite"] or r.get("non_finite_x_s", 0):
                problems.append(f"{m} chunk {c}: non-finite values")
            if m == "gmmd" and r["gmmd_fit"]["n_failed"]:
                problems.append(f"gmmd chunk {c}: {r['gmmd_fit']['n_failed']} IBW fits stopped early")
    mp = out_dir(cfg) / f"metrics_{a.tag}.json"
    if not mp.exists():
        problems.append(f"{mp.name} missing")
    else:
        mt = json.loads(mp.read_text())
        for m in METHODS:
            if not np.isfinite(mt["methods"][m]["fid"]):
                problems.append(f"{m}: FID not finite")
    for p in problems:
        log(f"CHECK FAILED: {p}")
    if problems:
        sys.exit(1)
    log(f"check passed for {a.n_chunks} chunks ({a.tag})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["prefix", "method", "e2e", "evaluate", "table", "check"])
    ap.add_argument("--config", default="exp4_interval_replacement")
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--n-chunks", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, parse_overrides(a.set))
    if a.n_chunks is None:
        a.n_chunks = int(cfg["generation"]["n_images"]) // int(cfg["generation"]["chunk"])
    dict(prefix=stage_prefix, method=stage_method, e2e=stage_e2e, evaluate=stage_evaluate,
         table=stage_table, check=stage_check)[a.stage](cfg, a, a.device)


if __name__ == "__main__":
    main()
