"""Provenance of the vendored ``score_sde_pytorch`` files, and the tool that (re)vendors them.

The teacher in :mod:`gmmvi.gmmd.teachers.score_sde` runs Yang Song et al.'s official PyTorch
implementation of *Score-Based Generative Modeling through Stochastic Differential Equations*
(https://github.com/yang-song/score_sde_pytorch, Apache-2.0, see ``LICENSE`` next to this
file).  The files under this directory are copies of the upstream files at
:data:`UPSTREAM_COMMIT`, altered ONLY by the substitutions declared in :data:`SUBSTITUTIONS`:

* absolute imports (``import sde_lib``, ``from models import ...``, ``from op import ...``)
  become relative ones, so the copy imports from inside this package;
* ``op/upfirdn2d.py`` keeps only upstream's pure-PyTorch ``upfirdn2d_native`` path (the branch
  upstream itself takes on CPU) and never builds the CUDA extension -- this machine has no
  ``nvcc``, and the native path is the same FIR filter evaluated with ``F.conv2d``;
* ``op/__init__.py`` drops ``fused_act`` (a second CUDA extension that NCSN++ never calls);
* ``sampling.py`` draws its Gaussian noise through a module-level ``randn_like`` hook that
  defaults to ``torch.randn_like``, so the teacher can install per-rollout generators without
  touching the update rules.

Nothing else differs, and ``tests/test_gmmd_vendor_parity.py`` re-downloads upstream to prove it.
Re-vendor from a fresh clone with::

    python -m gmmvi.gmmd.teachers.score_sde._vendor.provenance /path/to/score_sde_pytorch
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

UPSTREAM_REPO = "https://github.com/yang-song/score_sde_pytorch"
UPSTREAM_COMMIT = "cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44"
RAW = f"https://raw.githubusercontent.com/yang-song/score_sde_pytorch/{UPSTREAM_COMMIT}/"

# sha256 of the RAW BYTES of the upstream files at UPSTREAM_COMMIT (``sha256sum`` values).
# op/upfirdn2d.py is CRLF-terminated upstream; the vendored copy is normalised to LF (declared).
UPSTREAM_SHA = {
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "sde_lib.py": "6f1e755bc75a9a059821117b277688297b985f39d75122949592cecb0cf6c503",
    "sampling.py": "7254648febdf0a453aa9604b8ad8f1661337c2cd8fc9a1be17aa1b4f9a9afe59",
    "models/__init__.py": "b4d83f5cadbd1300480ce723b2bb572a7f5fff13017473bfd8d5c95969fa347f",
    "models/utils.py": "479479f872375f34494d1bd31725af744f9e7d2a11beac04b481801894ce379f",
    "models/ema.py": "de51eed0b88ee48347c3dc5db2321cc6b999dbcd895af870b096f27512465243",
    "models/ncsnpp.py": "d0698237c5fc66bbaf16e4dafafc1672d366e19a47ff88a025622d6ad0584f6d",
    "models/layerspp.py": "55714531661209ede4c30df64f8660abcf21addc09d6ccf947911dd7e8e8da50",
    "models/layers.py": "132b0c564d96c23448c2026cc1b19b8cafc484751f4a34a78ca56a25d01af7b5",
    "models/normalization.py": "0195a2f50be2df649e823a6a2bda91090d54f6c7ab480bbda99376898395c447",
    "models/up_or_down_sampling.py": "c7f3389dd9b9e24808ba99dfc583a966e71dd78b695acf9aa66d810ef6374b07",
    "op/__init__.py": "d3bdebb6b1ea35ab8e7dd14af32919f11ae630ab9154d4c84f839397b98b2bf6",
    "op/upfirdn2d.py": "e8902a530928ca5b7feb079d19933d2d15fc0afa6b2e36ed808c42ba83b1c88a",
}

# Vendored path -> list of edits applied to the upstream text, in order.  An edit is either
# ("replace", old, new) -- ``old`` must occur EXACTLY ``count`` times (default 1) -- or
# ("delete_lines", first, last) on the ORIGINAL 1-based line numbering (applied first).
_RANDN_HOOK = (
    "from .models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn\n"
    "from scipy import integrate\n"
    "from . import sde_lib\n"
    "from .models import utils as mutils\n"
    "\n"
    "# GMMD (declared substitution, see _vendor/provenance.py): every predictor / corrector draws\n"
    "# its Gaussian noise through this module-level hook instead of torch.randn_like, so a caller\n"
    "# can install per-rollout generators.  The default is exactly upstream's behaviour.\n"
    "randn_like = torch.randn_like\n"
)
SUBSTITUTIONS = {
    "sde_lib.py": [],
    "sampling.py": [
        ("replace",
         "from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn\n"
         "from scipy import integrate\n"
         "import sde_lib\n"
         "from models import utils as mutils\n",
         _RANDN_HOOK),
        ("replace", "torch.randn_like(x)", "randn_like(x)", 6),
    ],
    "models/__init__.py": [],
    "models/utils.py": [("replace", "import sde_lib\n", "from .. import sde_lib\n")],
    "models/ema.py": [],
    "models/ncsnpp.py": [],
    "models/layerspp.py": [],
    "models/layers.py": [],
    "models/normalization.py": [],
    "models/up_or_down_sampling.py": [
        ("replace", "from op import upfirdn2d\n", "from ..op import upfirdn2d\n")],
    "op/__init__.py": [
        ("replace", "from .fused_act import FusedLeakyReLU, fused_leaky_relu\n",
         "# GMMD: fused_act (a CUDA extension NCSN++ never calls) is not vendored.\n")],
    "op/upfirdn2d.py": [
        # the two autograd Functions that wrap the CUDA kernel (never reached below)
        ("delete_lines", 19, 143),
        ("replace",
         "from torch.utils.cpp_extension import load\n"
         "\n"
         "\n"
         "module_path = os.path.dirname(__file__)\n"
         "upfirdn2d_op = load(\n"
         "    \"upfirdn2d\",\n"
         "    sources=[\n"
         "        os.path.join(module_path, \"upfirdn2d.cpp\"),\n"
         "        os.path.join(module_path, \"upfirdn2d_kernel.cu\"),\n"
         "    ],\n"
         ")\n",
         "# GMMD (declared substitution, see _vendor/provenance.py): the CUDA extension is not\n"
         "# built; upfirdn2d() below always takes upstream's pure-PyTorch upfirdn2d_native path.\n"),
        ("replace",
         "def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):\n"
         "    if input.device.type == \"cpu\":\n"
         "        out = upfirdn2d_native(\n"
         "            input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]\n"
         "        )\n"
         "\n"
         "    else:\n"
         "        out = UpFirDn2d.apply(\n"
         "            input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])\n"
         "        )\n"
         "\n"
         "    return out\n",
         "def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):\n"
         "    # GMMD: native path on every device (upstream: CPU only; CUDA kernel otherwise).\n"
         "    out = upfirdn2d_native(\n"
         "        input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]\n"
         "    )\n"
         "\n"
         "    return out\n"),
    ],
    "LICENSE": [],
}

VENDOR_DIR = Path(__file__).resolve().parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def upstream_text(data: bytes) -> str:
    """Upstream bytes -> text with LF newlines (the one global normalisation, declared above)."""
    return data.decode("utf-8").replace("\r\n", "\n")


def apply_edits(text: str, edits) -> str:
    """Apply the declared edits to upstream ``text``; raises if an edit does not match."""
    deletes = sorted((e for e in edits if e[0] == "delete_lines"), key=lambda e: -e[1])
    lines = text.split("\n")
    for _, first, last in deletes:
        del lines[first - 1:last]
    text = "\n".join(lines)
    for e in edits:
        if e[0] != "replace":
            continue
        old, new = e[1], e[2]
        count = e[3] if len(e) > 3 else 1
        if text.count(old) != count:
            raise ValueError(f"substitution expected {count}x, found {text.count(old)}x: {old[:60]!r}")
        text = text.replace(old, new)
    return text


def revendor(upstream_dir):
    """Copy the upstream files at ``upstream_dir`` into this directory, applying the edits."""
    upstream_dir = Path(upstream_dir)
    for rel, edits in SUBSTITUTIONS.items():
        raw = (upstream_dir / rel).read_bytes()
        h = sha256(raw)
        if UPSTREAM_SHA[rel] != h:
            raise ValueError(f"{rel}: upstream sha {h} != recorded {UPSTREAM_SHA[rel]}; "
                             "review the diff by hand, then update UPSTREAM_SHA")
        dst = VENDOR_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(apply_edits(upstream_text(raw), edits))
        print(f"vendored {rel:32s} upstream sha256 {h[:12]}  ({len(edits)} edits)")


if __name__ == "__main__":
    revendor(sys.argv[1])
