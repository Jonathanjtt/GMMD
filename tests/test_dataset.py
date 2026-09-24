"""CelebA-HQ 256 loading / preprocessing: the checkpoint's exact input convention."""

import io

import numpy as np
import pytest
from PIL import Image

from gmmd.datasets import celebahq as ds


def _synthetic(size=256):
    a = np.zeros((size, size, 3), np.uint8)
    a[0, 0] = (255, 0, 0)          # red pixel top-left
    a[0, 1] = (0, 255, 0)          # green next to it
    a[1, 0] = (0, 0, 255)          # blue below it
    a[..., 0] += np.linspace(0, 200, size, dtype=np.uint8)[None, :]
    return a


def test_preprocess_matches_the_teacher_convention():
    a = _synthetic()
    x = ds.preprocess(a)
    assert x.shape == (3, 256, 256) and x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0                     # [0, 1], not [-1, 1]
    assert x[0, 0, 0] == 1.0 and x[1, 0, 1] == 1.0 and x[2, 1, 0] == 1.0   # CHW, RGB order
    assert np.array_equal(ds.to_uint8(x), a)                     # lossless round trip
    with pytest.raises(ValueError):
        ds.preprocess(a[:128])                                   # no silent resizing
    with pytest.raises(TypeError):
        ds.preprocess(a.astype(np.float32))


def test_image_dir_source(tmp_path):
    a = _synthetic()
    (tmp_path / "validation").mkdir()
    Image.fromarray(a).save(tmp_path / "validation" / "img_000.png")
    d = ds.CelebAHQ256(root=tmp_path, source="image_dir", split="validation")
    assert len(d) == 1 and d.identifier(0) == "img_000.png"
    assert np.array_equal(d.raw(0), a)
    assert np.array_equal(d[0], ds.preprocess(a))
    with pytest.raises(IndexError):
        d[1]


def test_hf_parquet_source(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    imgs = [_synthetic(), np.roll(_synthetic(), 5, axis=1)]
    rows = []
    for a in imgs:
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="PNG")
        rows.append({"bytes": buf.getvalue(), "path": None})
    tbl = pa.table({"image": pa.array(rows, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                    "label": pa.array([0, 1])})
    pq.write_table(tbl, tmp_path / "validation-00000-of-00001.parquet", row_group_size=1)
    d = ds.CelebAHQ256(root=tmp_path, source="hf_parquet", split="validation")
    assert len(d) == 2
    assert d.identifier(1) == "validation-00000-of-00001.parquet#row1"
    assert np.array_equal(d.raw(0), imgs[0]) and np.array_equal(d.raw(1), imgs[1])
    assert d[1].shape == (3, 256, 256)
    desc = d.describe()
    assert desc["centered"] is False and desc["layout"] == "CHW" and desc["n_images"] == 2
