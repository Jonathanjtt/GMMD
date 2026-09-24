"""Image -> feature vector, for the diagnostics of a conditional sample cloud.

Every extractor maps a batch of float CHW images in the teacher's data scale ([0, 1]) to an
``(N, D)`` float32 array.  ``"pixel"`` needs nothing (average-pooled pixels; the baseline that
is always available).  The pretrained encoders are opt-in because each pulls weights from the
internet on first use and one needs an extra package:

* ``"dinov2_vitb14"``  -- DINOv2 ViT-B/14 CLS embedding (768-d) via ``torch.hub``;
* ``"inception_pool3"`` -- Inception-v3 pool3 (2048-d, the FID feature) via ``torchvision``.

Which one the experiment uses is a config choice (``[features] extractor``); the raw feature
vectors are saved either way so the diagnostics can be re-run on another representation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["EXTRACTORS", "extract", "describe"]


def _as_images(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4:
        raise ValueError(f"expected (N, C, H, W) images, got {x.shape}")
    return x


def pixel_features(x, downsample=8, **_):
    """Average-pool by ``downsample`` and flatten: (N, C*H/f*W/f)."""
    x = _as_images(x)
    n, c, h, w = x.shape
    f = int(downsample)
    if f > 1:
        x = x[:, :, :h - h % f, :w - w % f].reshape(n, c, h // f, f, w // f, f).mean(axis=(3, 5))
    return x.reshape(n, -1).astype(np.float32)


def _imagenet_batches(x, size, device, batch_size):
    import torch
    import torch.nn.functional as F
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for i in range(0, len(x), batch_size):
        b = torch.as_tensor(x[i:i + batch_size]).to(device).clamp_(0.0, 1.0)
        b = F.interpolate(b, size=(size, size), mode="bicubic", align_corners=False)
        yield (b - mean) / std


def dinov2_features(x, device="cuda", batch_size=32, model_name="dinov2_vitb14", **_):
    """DINOv2 CLS token; weights fetched by torch.hub (facebookresearch/dinov2) on first use."""
    import torch
    x = _as_images(x)
    model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for b in _imagenet_batches(x, 224, device, batch_size):
            out.append(model(b).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def inception_features(x, device="cuda", batch_size=32, **_):
    """Inception-v3 pool3 features (the FID representation), via torchvision."""
    import torch
    from torchvision.models import inception_v3, Inception_V3_Weights
    x = _as_images(x)
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for b in _imagenet_batches(x, 299, device, batch_size):
            out.append(model(b).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


EXTRACTORS = {
    "pixel": pixel_features,
    "dinov2_vitb14": dinov2_features,
    "inception_pool3": inception_features,
}


def extract(name, images, **kw):
    if name not in EXTRACTORS:
        raise KeyError(f"unknown feature extractor {name!r}; available: {sorted(EXTRACTORS)}")
    return EXTRACTORS[name](images, **kw)


def describe(name, **kw):
    d = dict(extractor=name)
    if name == "pixel":
        d["downsample"] = int(kw.get("downsample", 8))
    return d
