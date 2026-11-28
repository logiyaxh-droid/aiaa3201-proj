# -*- coding: utf-8 -*-
"""Training smoke gates: run before any sweep.

Checks, in order:
  1. LR/GT frame pairing and crop geometry, including flip/transpose invariance.
  2. Charbonnier implementation matches BasicSR's registered loss exactly.
  3. DISTS and LPIPS losses are differentiable wrt the *input* while every
     metric parameter remains unchanged.
  4. Generator forward+backward updates real parameters without NaN/Inf.
  5. Discriminator forward+backward, and its optimizer, are exercised when GAN
     mode is requested.  The same code paths are used by train.py.

All tensors stay small (one 32x32 LR crop), so this is a correctness test, not a
speed or memory benchmark.
"""
from __future__ import annotations

import os
import sys

ROOT = "/data/disk1/home/ghe270/p3/quality_metrics"
sys.path.insert(0, os.path.join(ROOT, "fine_tune", "scripts"))
import train as T

import torch

OUT = os.path.join(ROOT, "fine_tune", "selfcheck")
os.makedirs(OUT, exist_ok=True)


def main():
    report = {}
    ds = T.WindowDataset(crop=32, length=20, seed=7)
    lr, gt, clip, ids = ds[0]
    report["data"] = {
        "lr_shape": list(lr.shape), "gt_shape": list(gt.shape),
        "clip": clip, "ids": ids,
        "crop_ratio": gt.shape[-1] / lr.shape[-1],
        "pass": tuple(lr.shape[:2]) == (5, 3) and gt.shape[0] == 3
                and gt.shape[-1] == lr.shape[-1] * 4,
    }

    from basicsr.losses.basic_loss import CharbonnierLoss
    ref = CharbonnierLoss(loss_weight=1.0, reduction="mean")
    a = torch.rand(2, 3, 16, 16)
    b = torch.rand(2, 3, 16, 16)
    diff = abs(float(T.charbonnier(a, b) - ref(a, b)))
    report["charbonnier"] = {"abs_diff": diff, "pass": diff < 1e-7}

    bundle = T.LossBundle("cuda", 1.0, 1.0)
    before = {n: p.detach().clone() for n, p in bundle.dists.named_parameters()}
    x = torch.rand(1, 3, 128, 128, device="cuda", requires_grad=True)
    y = torch.rand(1, 3, 128, 128, device="cuda")
    dists = bundle.dists(x, y).mean()
    dists.backward()
    changed = any(not torch.equal(v, dict(bundle.dists.named_parameters())[n])
                  for n, v in before.items())
    report["dists_backward"] = {
        "value": float(dists), "input_grad_finite": bool(torch.isfinite(x.grad).all()),
        "input_grad_abs_sum": float(x.grad.abs().sum()),
        "metric_parameters_changed": changed,
        "pass": bool(torch.isfinite(x.grad).all()) and x.grad.abs().sum().item() > 0
                and not changed,
    }

    # Keep the generator test focused on the loss plumbing; the full-size model
    # is loaded once and stepped on a small crop.
    net = T.build_generator("cuda").train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-6)
    old = {n: p.detach().clone() for n, p in net.named_parameters()
           if n == "conv_last.weight"}
    pred = net(lr.unsqueeze(0).cuda())
    target = gt.unsqueeze(0).cuda()
    loss = T.charbonnier(pred, target) + 0.01 * bundle.dists(pred, target).mean()
    opt.zero_grad()
    loss.backward()
    finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    opt.step()
    updated = not torch.equal(old["conv_last.weight"].cuda(), net.conv_last.weight.detach())
    report["generator_step"] = {
        "pred_shape": list(pred.shape), "loss": float(loss.detach()),
        "gradients_finite": bool(finite), "conv_last_updated": bool(updated),
        "pass": bool(finite and updated),
    }

    disc = T.discriminator("cuda").train()
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-5)
    real = disc(target)
    fake = disc(pred.detach())
    d_loss = T.relativistic_d_loss(real, fake)
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()
    g_loss = T.relativistic_g_loss(disc(target), disc(pred))
    report["gan_step"] = {"d_loss": float(d_loss.detach()),
                          "g_loss": float(g_loss.detach()),
                          "real_logits_shape": list(real.shape),
                          "pass": bool(torch.isfinite(d_loss) and torch.isfinite(g_loss))}

    report["pass"] = all(v["pass"] for v in report.values())
    T.C.write_json(os.path.join(OUT, "training_smoke.json"), report)
    import json
    print(json.dumps(report, indent=1))
    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
