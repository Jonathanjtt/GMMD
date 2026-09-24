"""CelebA-HQ 256x256 images, preprocessed exactly as ``score_sde_pytorch/datasets.py`` does.

What the teacher was trained on (``datasets.get_dataset`` branch ``'CelebAHQ'``): the
Progressive-GAN ``r08.tfrecords`` of 256x256 uint8 images, decoded, transposed to HWC,
``tf.image.convert_image_dtype(., tf.float32)`` -> values in [0, 1], random left-right flips
during training, no other transform; ``data.centered = False`` so NO rescaling to [-1, 1]
(the NCSN++ network centres its input itself).  The network takes CHW, RGB.  So an image
reaches the teacher as a float32 (3, 256, 256) RGB tensor in [0, 1], and that is what
:func:`preprocess` produces from a uint8 HWC RGB array.

Sources.  Upstream's tfrecords are not distributed; this loader reads either a directory of
256x256 image files or a Hugging Face parquet shard (``korexyz/celeba-hq-256x256``: the 30 000
CelebA-HQ images as lossless 256x256 PNGs, 28 000 / 2 000 train / validation).  One caveat is
recorded rather than hidden: how that mirror produced its 256x256 PNGs from the 1024x1024
originals is not documented, while the Progressive-GAN tfrecord tool box-filters (4x4 mean) --
a resampling-kernel difference that is invisible next to the sigma >= 0.01 noise the teacher
always sees, but not a bit-exact match.  Also: the teacher was trained on ALL 30 000 images
(upstream uses one 'train' split for train and eval), so no CelebA-HQ image is held out from
the checkpoint's training set; "held-out" below means held out of THIS experiment's fitting,
not of the teacher's training.  Both points are stated in the experiment's provenance.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np

__all__ = ["preprocess", "to_uint8", "CelebAHQ256", "data_root", "DEFAULT_DATA_ROOT"]

IMAGE_SIZE = 256
CHANNELS = 3
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# <repo>/data/ by default (gitignored: 1.2 GB of checkpoint + shards), overridable by env var.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def data_root(root=None):
    """The data directory: explicit ``root`` > ``$GMMD_DATA_DIR`` > ``<repo>/data``."""
    if root is not None:
        return Path(root).expanduser()
    return Path(os.environ.get("GMMD_DATA_DIR", DEFAULT_DATA_ROOT)).expanduser()


def preprocess(img_u8, image_size=IMAGE_SIZE):
    """uint8 HWC RGB (H, W, 3) -> float32 CHW in [0, 1]: the checkpoint's input convention."""
    a = np.asarray(img_u8)
    if a.dtype != np.uint8:
        raise TypeError(f"expected uint8 image, got {a.dtype}")
    if a.ndim != 3 or a.shape[2] != CHANNELS:
        raise ValueError(f"expected an (H, W, 3) RGB image, got {a.shape}")
    if a.shape[0] != image_size or a.shape[1] != image_size:
        raise ValueError(f"expected a {image_size}x{image_size} image, got {a.shape[:2]}; this "
                         "loader does not resize, the teacher's training images were already "
                         f"{image_size}x{image_size}")
    x = a.astype(np.float32) / 255.0                 # tf.image.convert_image_dtype(uint8->f32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))  # HWC -> CHW


def to_uint8(x_chw):
    """Inverse for display: float CHW (any range; clipped to [0, 1]) -> uint8 HWC."""
    x = np.asarray(x_chw, dtype=np.float32)
    if x.ndim == 4:
        return np.stack([to_uint8(v) for v in x])
    x = np.clip(x, 0.0, 1.0).transpose(1, 2, 0)
    return np.round(x * 255.0).astype(np.uint8)


class CelebAHQ256:
    """Index-addressable CelebA-HQ 256 images from a parquet shard or an image directory.

    ``source="hf_parquet"``: every ``<split>-*.parquet`` under ``root`` (HF ``Image`` feature:
    PNG bytes per row), rows in file order.  ``source="image_dir"``: every image file under
    ``root/<split>`` (or ``root`` if that folder does not exist), sorted by name.  ``ids`` are
    positions in that order; :meth:`identifier` gives the stable name recorded in provenance.
    """

    def __init__(self, root=None, source="hf_parquet", split="validation", image_size=IMAGE_SIZE):
        self.root = data_root(root) if source == "hf_parquet" and root is None \
            else Path(root).expanduser() if root is not None else data_root()
        self.source = source
        self.split = split
        self.image_size = int(image_size)
        if source == "hf_parquet":
            self._files = sorted(self.root.glob(f"{split}-*.parquet"))
            if not self._files:
                raise FileNotFoundError(f"no {split}-*.parquet under {self.root}")
            import pyarrow.parquet as pq
            self._pq = pq
            self._offsets = [0]
            for f in self._files:
                self._offsets.append(self._offsets[-1] + pq.ParquetFile(f).metadata.num_rows)
            self._n = self._offsets[-1]
        elif source == "image_dir":
            d = self.root / split
            d = d if d.is_dir() else self.root
            self._files = sorted(p for p in d.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
            if not self._files:
                raise FileNotFoundError(f"no image files under {d}")
            self._n = len(self._files)
        else:
            raise ValueError(f"unknown source {source!r}; use 'hf_parquet' or 'image_dir'")

    def __len__(self):
        return self._n

    def identifier(self, idx):
        """A stable per-image name: ``<split>/<file>#<row>`` or the image file name."""
        idx = self._check(idx)
        if self.source == "hf_parquet":
            k = int(np.searchsorted(self._offsets, idx, side="right") - 1)
            return f"{self._files[k].name}#row{idx - self._offsets[k]}"
        return self._files[idx].name

    def _check(self, idx):
        idx = int(idx)
        if not 0 <= idx < self._n:
            raise IndexError(f"image index {idx} out of range [0, {self._n})")
        return idx

    def raw(self, idx):
        """The uint8 HWC RGB array of image ``idx`` (decoded, no preprocessing)."""
        from PIL import Image
        idx = self._check(idx)
        if self.source == "hf_parquet":
            k = int(np.searchsorted(self._offsets, idx, side="right") - 1)
            row = idx - self._offsets[k]
            pf = self._pq.ParquetFile(self._files[k])
            # walk row groups to the one holding `row` (shards are small; no need for an index)
            start = 0
            for g in range(pf.metadata.num_row_groups):
                n = pf.metadata.row_group(g).num_rows
                if row < start + n:
                    tbl = pf.read_row_group(g, columns=["image"])
                    payload = tbl.column("image")[row - start].as_py()["bytes"]
                    break
                start += n
            im = Image.open(io.BytesIO(payload))
        else:
            im = Image.open(self._files[idx])
        return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def __getitem__(self, idx):
        """float32 CHW image in [0, 1], the teacher's convention."""
        return preprocess(self.raw(idx), self.image_size)

    def describe(self):
        return dict(name="celebahq256", source=self.source, root=str(self.root), split=self.split,
                    n_images=int(self._n), image_size=self.image_size, centered=False,
                    value_range=[0.0, 1.0], layout="CHW", channel_order="RGB",
                    files=[f.name for f in self._files] if self.source == "hf_parquet"
                    else f"{self._n} image files")
