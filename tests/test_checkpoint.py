"""Integration: the real CelebA-HQ 256 checkpoint on a GPU.  Skipped when either is absent.

Run with   CUDA_VISIBLE_DEVICES=<free gpu> python -m pytest tests/test_checkpoint.py -m checkpoint
"""

import numpy as np
import pytest
import torch

from gmmd.datasets.celebahq import CelebAHQ256, data_root
from gmmd.download import CHECKPOINTS

pytestmark = pytest.mark.checkpoint

NAME = "ve/celebahq_256_ncsnpp_continuous"
CKPT = data_root() / "checkpoints" / NAME / CHECKPOINTS[NAME]["file"]


@pytest.fixture(scope="module")
def teacher():
    if not CKPT.exists():
        pytest.skip(f"checkpoint not present: {CKPT}")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from gmmd.teachers.score_sde.teacher import ScoreSDETeacher
    return ScoreSDETeacher.from_config(NAME, CKPT, device="cuda")


@pytest.fixture(scope="module")
def x0():
    try:
        return torch.as_tensor(CelebAHQ256(root=data_root() / "celebahq256", split="validation")[0])
    except FileNotFoundError:
        pytest.skip("CelebA-HQ validation shard not present")


def _psnr(a, b):
    return 10 * np.log10(1.0 / float(((a - b) ** 2).mean()))


def test_checkpoint_loads_with_ema_and_official_sde(teacher):
    assert teacher.info["step"] == 2400005 and teacher.info["use_ema"]
    assert teacher.sde.N == 2000 and teacher.sde.sigma_max == 348 and teacher.eps == 1e-5
    assert teacher.image_shape == (3, 256, 256)


def test_preprocessing_is_compatible_with_the_teacher(teacher, x0):
    """Tweedie denoising of a lightly noised dataset image must recover it well."""
    x_t, info = teacher.forward_noise(x0, 0.25, seed=0)
    xhat = teacher.tweedie_denoise(x_t, info["t"])[0].cpu()
    assert _psnr(xhat.numpy(), x0.numpy()) > 30.0
    # the wrong value range ([-1, 1]) is visibly worse: the check discriminates
    bad = 2 * x0 - 1
    xb, _ = teacher.forward_noise(bad, 0.25, seed=0)
    assert _psnr(teacher.tweedie_denoise(xb, info["t"])[0].cpu().numpy(), bad.numpy()) < 27.0


def test_rollouts_reproducible_and_stochastic_on_gpu(teacher, x0):
    x_t, _ = teacher.forward_noise(x0, 0.6, seed=0)
    a = teacher.sample_transition(x_t, 0.6, 0.597, seeds=(1, 2))
    b = teacher.sample_transition(x_t, 0.6, 0.597, seeds=(1, 2))
    c = teacher.sample_transition(x_t, 0.6, 0.597, seeds=(3, 2))
    assert torch.equal(a.x_s, b.x_s) and not torch.equal(a.x_s[0], c.x_s[0])
    assert a.nfe_per_sample == a.n_steps == 6
    assert torch.isfinite(a.x_s).all()
    d1 = teacher.ode_transition(x_t, 0.6, 0.597)
    d2 = teacher.ode_transition(x_t, 0.6, 0.597)
    assert float((d1.x_s - d2.x_s).abs().max()) < 1e-4
