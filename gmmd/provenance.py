"""What every GMMD results directory records so a run can be traced and regenerated.

``snapshot()`` collects the git commit (and whether the tree was dirty), the resolved config,
package versions, the host and GPU, and the wall clock; the experiment adds the teacher
description (checkpoint digest, SDE, grid), the dataset description, the image identifier,
every seed, ``t`` / ``s``, sample counts, NFE and runtimes as it goes.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

__all__ = ["git_state", "package_versions", "device_info", "snapshot", "dump_json"]

_REPO = Path(__file__).resolve().parents[1]
_PACKAGES = ("numpy", "scipy", "torch", "jax", "jaxlib", "sklearn", "umap", "matplotlib",
             "PIL", "pyarrow", "einops", "torchvision")


def git_state(repo=_REPO):
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:  # not a git checkout, git missing, ...
            return ""
    commit = run("rev-parse", "HEAD")
    return dict(commit=commit or None, branch=run("rev-parse", "--abbrev-ref", "HEAD") or None,
                dirty=bool(run("status", "--porcelain")) if commit else None)


def package_versions(names=_PACKAGES):
    out = {"python": sys.version.split()[0]}
    for n in names:
        try:
            out[n] = getattr(importlib.import_module(n), "__version__", "?")
        except Exception:
            out[n] = None
    return out


def device_info(device=None):
    info = dict(hostname=socket.gethostname(), platform=platform.platform())
    try:
        import torch
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            idx = torch.device(device).index if device is not None and \
                torch.device(device).type == "cuda" else torch.cuda.current_device()
            idx = 0 if idx is None else idx
            p = torch.cuda.get_device_properties(idx)
            info["gpu"] = dict(name=p.name, index=idx, total_memory_gb=round(p.total_memory / 2**30, 1),
                               cudnn_deterministic=torch.backends.cudnn.deterministic,
                               allow_tf32=torch.backends.cudnn.allow_tf32)
    except Exception:
        pass
    return info


def snapshot(config, device=None):
    return dict(timestamp=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                git=git_state(), config=config, packages=package_versions(),
                machine=device_info(device), argv=sys.argv)


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (Path,)):
        return str(v)
    if isinstance(v, (set, tuple)):
        return list(v)
    return str(v)


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))
    return path
