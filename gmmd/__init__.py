"""GMMD -- is the stochastic diffusion transition ``p_theta(x_s | x_t)`` multimodal?

Two questions, in order:

**H1.** For a pretrained score-based diffusion model and ONE fixed noisy image ``x_t``, does
the stochastic reverse-SDE transition to an earlier time ``s`` produce a genuinely multimodal
conditional ``p_theta(x_s | x_t)``, or one broad mode?  This has to be separated from the
trivial fact that the image dataset itself is multimodal, which is why every rollout starts
from the same frozen tensor and the experiment carries a deterministic control.

**H2 (later).** If it is multimodal, can a mixture of isotropic Gaussians fitted by IBW
model that conditional better than a single Gaussian?

Layout::

    gmmd/config.py         configs/*.toml loader (inherit = "<base>" variants)
    gmmd/provenance.py     git commit, package versions, GPU, checkpoint digest -> json
    gmmd/download.py       fetch the teacher checkpoint and the CelebA-HQ shards
    gmmd/teachers/         the stochastic teacher: Song et al.'s score SDE, vendored
    gmmd/datasets/         CelebA-HQ 256 with the teacher checkpoint's exact preprocessing
    gmmd/features.py       image -> feature vector (pixel / DINOv2 / Inception)
    gmmd/diagnostics.py    multimodality DIAGNOSTICS of a conditional sample cloud
    gmmd/plotting.py       sample grids, embedding panels, model-selection figures
    gmmd/ibw.py            Petit-Talamon, Lambert & Korba's Algorithm 1 (IBW / MD / NGD)
    gmmd/vi.py             targets for it -- and why the teacher conditional is not one yet
    gmmd/experiments/      conditional_multimodality: the H1 experiment

Nothing is re-exported here on purpose: importing ``gmmd`` must not pull in torch, so
``gmmd.ibw`` and ``gmmd.diagnostics`` stay usable in a plain-numpy session.
"""
