"""The standalone amortized IBW-GMMD sampler: ten conditional-network evaluations and ten
Gaussian-mixture draws per image, from a student checkpoint alone.

    python -m gmmd.exp5.sampler --ckpt outputs/exp5_ibw_amortized_10step_k8/student/main/ckpt_best.pt \
        --n 64 --seed-base 1000000 --out /path/samples.npz [--precision tf32|fp32] [--batch-size 64]

Per image:

1. ``x_T = sigma_max z`` with ``z`` from a CPU generator seeded by the image seed -- the teacher's
   prior, bit-identical to :func:`gmmd.pipeline.prior_sample`;
2. for each interval ``j = 0..9``: ONE network evaluation ``(x, j) -> (pi, mu_1..K, v_1..K)``,
   ONE component index ``k ~ Categorical(pi)``, then ``x <- mu_k + sqrt(v_k) eps`` with fresh
   ``eps ~ N(0, I)``; the draws come from the image's own generator, so an image does not depend
   on its batch;
3. the last draw is the final image (the teacher's output representation, float, ~[0, 1]).

No teacher is constructed and no teacher score, IBW iteration, optimizer step or backward pass
can run: the loop executes under ``torch.inference_mode``, and :meth:`StudentSampler.sample`
returns an instrumentation record proving the counts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gmmd import ibw_batched
from gmmd.exp5.grid import SeededGenerators, prior_from_seeds, seed_of
from gmmd.exp5.student import MixtureStudent, count_parameters
from gmmd.teachers.score_sde import teacher as teacher_mod

__all__ = ["StudentSampler", "main"]


class StudentSampler:
    def __init__(self, ckpt, device="cuda", precision="tf32", weights="ema", final_noise=True):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.ckpt_path = str(ckpt)
        self.meta = ck["meta"]
        self.net = MixtureStudent.from_arch(ck["arch"])
        self.net.load_state_dict(ck[weights])
        self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)
        self.precision = precision
        self.weights = weights
        self.n_intervals = self.net.n_intervals
        self.image_shape = tuple(self.net.image_shape)
        self.sigma_max = float(self.meta["sigma_max"])
        self.step = int(ck.get("step", -1))
        # final_noise=False is a post-hoc DIAGNOSTIC, not the method: the last transition returns the
        # drawn component's mean (same x_T, components and states as the regular sampler; the last
        # noise draw is made and discarded), to measure what the final draw's noise costs
        self.final_noise = bool(final_noise)

    def _set_precision(self):
        on = self.precision == "tf32"
        torch.backends.cudnn.allow_tf32 = on
        torch.backends.cuda.matmul.allow_tf32 = on

    def draw_seeds(self, seeds):
        return [seed_of(71, s) for s in seeds]

    @torch.inference_mode()
    def _batch(self, seeds, keep=False):
        x = prior_from_seeds(self.image_shape, self.sigma_max, seeds, self.device)
        gens = SeededGenerators(self.draw_seeds(seeds), self.device)
        B = len(seeds)
        comps, ents, states = [], [], [x.cpu()] if keep else None
        draws = 0
        for j in range(self.n_intervals):
            logits, means, logv = self.net(x, j)                  # ONE network evaluation
            p = torch.softmax(logits.float(), -1)
            ents.append((-(p * p.clamp_min(1e-30).log()).sum(-1)).cpu())
            k = torch.stack([torch.multinomial(p[b], 1, generator=gens.gens[b])[0] for b in range(B)])
            eps = gens.randn(self.image_shape)
            idx = torch.arange(B, device=x.device)
            sd = (0.5 * logv[idx, k]).exp().view(B, 1, 1, 1)
            if j == self.n_intervals - 1 and not self.final_noise:
                sd = torch.zeros_like(sd)                          # diagnostic only (see __init__)
            x = means[idx, k] + sd * eps                           # ONE GMM draw
            draws += B
            comps.append(k.cpu())
            if keep:
                states.append(x.cpu())
        return x, torch.stack(comps, 1), torch.stack(ents, 1), draws, states

    def sample(self, seeds, batch_size=64, keep_intermediates=False):
        """Images ``(n, C, H, W)`` float32 (CPU) and the instrumentation record."""
        self._set_precision()
        seeds = [int(s) for s in seeds]
        c0 = dict(net_images=self.net.n_forward, net_calls=self.net.n_calls,
                  teacher_scores=teacher_mod.SCORE_EXAMPLES[0], teachers=teacher_mod.INSTANCES[0],
                  ibw_iterations=ibw_batched.ITERATION_COUNTER[0])
        imgs, comps, ents, states = [], [], [], []
        draws = 0
        n_batches = 0
        for i in range(0, len(seeds), batch_size):
            x, k, e, nd, st = self._batch(seeds[i:i + batch_size], keep_intermediates)
            imgs.append(x.float().cpu())
            comps.append(k)
            ents.append(e)
            draws += nd
            n_batches += 1
            if keep_intermediates:
                states.append(torch.stack(st, 1))
        n = len(seeds)
        rec = dict(
            n_images=n, n_intervals=self.n_intervals, n_batches=n_batches,
            network_evals_per_image=(self.net.n_forward - c0["net_images"]) / n,
            network_calls=self.net.n_calls - c0["net_calls"],
            gmm_draws_per_image=draws / n,
            teacher_score_evals=teacher_mod.SCORE_EXAMPLES[0] - c0["teacher_scores"],
            teachers_constructed=teacher_mod.INSTANCES[0] - c0["teachers"],
            ibw_iterations=ibw_batched.ITERATION_COUNTER[0] - c0["ibw_iterations"],
            inference_mode=True, any_param_requires_grad=any(p.requires_grad for p in self.net.parameters()),
            any_param_grad=any(p.grad is not None for p in self.net.parameters()),
            optimizer_steps=0, precision=self.precision, weights=self.weights, final_noise=self.final_noise,
            student_parameters=count_parameters(self.net), checkpoint=self.ckpt_path, step=self.step)
        rec["verified"] = bool(rec["network_evals_per_image"] == self.n_intervals
                               and rec["gmm_draws_per_image"] == self.n_intervals
                               and rec["teacher_score_evals"] == 0 and rec["teachers_constructed"] == 0
                               and rec["ibw_iterations"] == 0 and not rec["any_param_requires_grad"]
                               and not rec["any_param_grad"])
        out = dict(images=torch.cat(imgs), components=torch.cat(comps), weight_entropy=torch.cat(ents))
        if keep_intermediates:
            out["states"] = torch.cat(states)
        return out, rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed-base", type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", choices=["tf32", "fp32"], default="tf32")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    s = StudentSampler(a.ckpt, device=a.device, precision=a.precision, weights=a.weights)
    seeds = list(range(a.seed_base, a.seed_base + a.n))
    t0 = time.perf_counter()
    out, rec = s.sample(seeds, batch_size=a.batch_size)
    rec["seconds_total"] = time.perf_counter() - t0
    from gmmd.fid import to_uint8_upstream
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, images=to_uint8_upstream(out["images"]), seeds=np.asarray(seeds),
             components=out["components"].numpy())
    Path(a.out).with_suffix(".json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    if not rec["verified"]:
        raise SystemExit("instrumentation check FAILED")


if __name__ == "__main__":
    main()
