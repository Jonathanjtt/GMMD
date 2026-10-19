"""Shared plumbing for the exp5 stages: paths, splits, seeds, the teacher, cached arrays."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from gmmd import provenance as Pv
from gmmd.config import resolve_path
from gmmd.datasets.celebahq import data_root
from gmmd.exp5.grid import make_partition, partition_table, seed_of

_REPO = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test", "refine")

# seed namespaces (first key of seed_of); one per kind of randomness
NS_REF, NS_INIT, NS_NOISE, NS_EVAL, NS_PRINCIPAL, NS_BANK, NS_ENDPOINT = 81, 82, 83, 84, 85, 86, 87


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def out_dir(cfg):
    return _REPO / cfg["output"]["dir"]


def build_teacher(cfg, device, tf32):
    from gmmd.teachers.score_sde.teacher import ScoreSDETeacher

    tc = cfg["teacher"]
    if tc["predictor"] != "reverse_diffusion" or tc["corrector"] != "none" or not tc["noise_removal"]:
        raise NotImplementedError("exp5 implements the predictor-only reverse-diffusion teacher "
                                  "with noise removal")
    return ScoreSDETeacher.from_config(tc["config"], resolve_path(tc["checkpoint"], data_root()),
                                       device=device, use_ema=bool(tc["use_ema"]),
                                       allow_tf32=bool(tf32))


def teacher_id(teacher):
    i = teacher.info
    return dict(config_name=i.get("config_name"), checkpoint=i.get("checkpoint"),
                checkpoint_sha256=i.get("checkpoint_sha256"), step=i.get("step"),
                use_ema=i.get("use_ema"), upstream_commit=i.get("upstream_commit"),
                sde=type(teacher.sde).__name__, N=int(teacher.sde.N), eps=teacher.eps,
                sampler="reverse_diffusion predictor, no corrector, noise removal")


def partition(cfg, teacher):
    return make_partition(teacher, int(cfg["partition"]["n_intervals"]))


def split_seeds(cfg, split):
    """[(trajectory index n, x_T seed)] for a split; refine anchors come from student rollouts."""
    s = cfg["splits"]
    base, n = int(s[f"{split}_base"]), int(s[f"n_{split}"])
    return [(k, base + k) for k in range(n)]


def sid(split):
    return SPLITS.index(split)


def ref_seeds(split, n, j, n_rollouts):
    return [seed_of(NS_REF, sid(split), n, j, r) for r in range(n_rollouts)]


def anchor_path(cfg, split, n):
    return out_dir(cfg) / "anchors" / split / f"traj{n:03d}.npz"


def load_anchor(cfg, split, n, j):
    """The start state of interval ``j`` on trajectory ``n`` of ``split``: (C, H, W) float32."""
    with np.load(anchor_path(cfg, split, n)) as z:
        return torch.as_tensor(z["states"][j])


def ref_path(cfg, split, n, j):
    return out_dir(cfg) / "refs" / split / f"traj{n:03d}_j{j}.npz"


def fit_path(cfg, split, variant, n, j):
    return out_dir(cfg) / "fits" / split / variant / f"traj{n:03d}_j{j}.npz"


def variant_cfg(cfg, name):
    for k, v in enumerate(cfg["ibw"]["variants"]):
        if v["name"] == name:
            return k, dict(v)
    if name.startswith("p0_c"):                    # P0 step-size check: prior-free init
        c = float(name[len("p0_c"):].replace("p", "."))
        return 100 + int(round(100 * c)), dict(name=name, scheme="prior_free", spread=1.0, var="gap",
                                                step_c=c)
    if name == "frozen":
        sel = json.loads((out_dir(cfg) / "selection.json").read_text())
        k, v = variant_cfg(cfg, sel["variant"])
        v = dict(v, name="frozen", source_variant=sel["variant"], step_c=sel["step_c"],
                 n_iterations=sel["iterations"])
        return k, v
    raise KeyError(f"unknown IBW variant {name!r}")


def save_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)


def dump(obj, path):
    return Pv.dump_json(obj, path)


def snapshot(cfg, device):
    return Pv.snapshot(cfg, device)


def git_state():
    return Pv.git_state()


def env_precision():
    return dict(cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32))


def partition_table_dicts(parts):
    return partition_table(parts)


# ======================================================================================
# dynamic work claims: several workers (one per GPU) can run the same stage on one queue
# ======================================================================================
def _pid_alive(pid):
    import os
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def claim(lock_dir):
    """Atomically claim a unit of work (``mkdir``).  A lock left by a dead process is reclaimed."""
    import os
    import shutil
    import socket

    lock_dir = Path(lock_dir)
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(lock_dir)
    except FileExistsError:
        try:
            host, pid = (lock_dir / "owner").read_text().split()
        except (OSError, ValueError):
            return False                               # being created right now
        if host != socket.gethostname() or _pid_alive(pid):
            return False
        shutil.rmtree(lock_dir, ignore_errors=True)    # stale: its owner died
        try:
            os.mkdir(lock_dir)
        except FileExistsError:
            return False
    (lock_dir / "owner").write_text(f"{socket.gethostname()} {os.getpid()}")
    return True


def release(lock_dir):
    import shutil
    shutil.rmtree(lock_dir, ignore_errors=True)


def lock_path(cfg, kind, split, *parts):
    return out_dir(cfg) / "locks" / kind / split / "_".join(str(p) for p in parts)
