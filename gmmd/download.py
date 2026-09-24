"""Fetch the GMMD data files: the teacher checkpoint (Google Drive) and CelebA-HQ 256 shards.

    python -m gmmd.download --checkpoint ve/celebahq_256_ncsnpp_continuous
    python -m gmmd.download --celebahq validation

Files land under the data root (``$GMMD_DATA_DIR`` or ``gmmd/data``) and are checked
against the recorded sha256 digests.  The checkpoint needs ``gdown`` (``pip install gdown``).
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from gmmd.datasets.celebahq import data_root

CHECKPOINTS = {
    # score_sde_pytorch README: "ve/celebahq_256_ncsnpp_continuous" -- the re-trained CelebA-HQ 256
    "ve/celebahq_256_ncsnpp_continuous": dict(
        gdrive_id="1ocvHVzAeYtwIRFPgqG1CPPHdXUzY85UG", file="checkpoint_48.pth",
        sha256="3abcb7d219a80b9929e8b5a93b184fa55c428063c33a21000266741b091bd424"),
}
_HF = "https://huggingface.co/datasets/korexyz/celeba-hq-256x256/resolve/main/data/"
CELEBAHQ_SHARDS = {
    "validation": [("validation-00000-of-00001.parquet",
                    "299f7c95911c747905cba7f1e46ed69bc98a18a6b3cc9384cbda4c77eabeedcf")],
    "train": [(f"train-0000{i}-of-00006.parquet", None) for i in range(6)],
}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _check(path, expected):
    if expected:
        got = sha256(path)
        if got != expected:
            raise RuntimeError(f"{path}: sha256 {got} != expected {expected}")
    return path


def fetch_checkpoint(name, root=None):
    spec = CHECKPOINTS[name]
    dst = data_root(root) / "checkpoints" / name / spec["file"]
    if dst.exists():
        return _check(dst, spec["sha256"])
    try:
        import gdown
    except ImportError as e:
        raise ImportError("downloading the checkpoint needs `pip install gdown`; or fetch "
                          f"https://drive.google.com/uc?id={spec['gdrive_id']} by hand to {dst}") from e
    dst.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=spec["gdrive_id"], output=str(dst), quiet=False)
    return _check(dst, spec["sha256"])


def fetch_celebahq(split="validation", root=None):
    d = data_root(root) / "celebahq256"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for fname, digest in CELEBAHQ_SHARDS[split]:
        dst = d / fname
        if not dst.exists():
            print(f"downloading {fname} ...")
            urllib.request.urlretrieve(_HF + fname, dst)
        out.append(_check(dst, digest))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=None, choices=sorted(CHECKPOINTS))
    ap.add_argument("--celebahq", default=None, choices=sorted(CELEBAHQ_SHARDS))
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    if a.checkpoint:
        print("checkpoint ->", fetch_checkpoint(a.checkpoint, a.root))
    if a.celebahq:
        print("shards ->", fetch_celebahq(a.celebahq, a.root))


if __name__ == "__main__":
    main()
