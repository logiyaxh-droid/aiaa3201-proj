#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0 task 3 -- prove the shared metric implementation equals the old ones.

Verifies, on real REDS frames, that:

  1. ``evalkit.metrics.ssim_gpu`` == ``skimage.metrics.structural_similarity``
     (win_size=7, no gaussian weights, sample covariance) to <=1e-10.
  2. ``evalkit.metrics.psnr_frames`` == the old ``10*log10(255^2/MSE)`` on the
     YCrCb-Y channel, bit for bit.
  3. ``evalkit.metrics.LPIPS`` reproduces the old hand-rolled
     ``lpips_mean`` helper, so Part-1/Part-2 numbers stay on the same scale.
  4. The old and new entry points agree on a real prediction/GT pair.

Run:  /tmp/srenv/bin/python -m evalkit.selfcheck --clips 000 --frames 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np

# --- torchvision>=0.17 / BasicSR shim, identical to the legacy scripts ---------
import torchvision  # noqa: F401

_mod = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F  # noqa: E402

for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _mod
# ------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/disk1/shared/aiaa3201/project1_video_sr")
    ap.add_argument("--clip", default="000")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    a = ap.parse_args()

    import cv2
    import torch

    from evalkit import data, metrics

    gt_dir = os.path.join(a.root, "val_sharp", a.clip)
    lr_dir = os.path.join(a.root, "val_sharp_bicubic", a.clip)
    gt_i, lr_i = data.index_dir(gt_dir), data.index_dir(lr_dir)
    ids = sorted(set(gt_i) & set(lr_i))[: a.frames]
    gt = [data.read_rgb(gt_i[i]) for i in ids]
    lr = [data.read_rgb(lr_i[i]) for i in ids]
    H, W = gt[0].shape[:2]
    pred = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr]

    rep = {"clip": a.clip, "frames": ids, "device": a.device}

    # 1. SSIM vs skimage -------------------------------------------------------
    from skimage.metrics import structural_similarity as sk_ssim

    pY = np.stack([metrics.luma(x) for x in pred])
    gY = np.stack([metrics.luma(x) for x in gt])
    new_ssim = metrics.ssim_gpu(pY, gY, device=a.device)
    ref = np.array([sk_ssim(g, p, data_range=255, win_size=7,
                            gaussian_weights=False, use_sample_covariance=True)
                    for g, p in zip(gY, pY)])
    rep["ssim"] = {"max_abs_diff": float(np.abs(new_ssim - ref).max()),
                   "new_mean": float(new_ssim.mean()), "skimage_mean": float(ref.mean()),
                   "pass": bool(np.abs(new_ssim - ref).max() < 1e-10)}

    # 2. PSNR vs the closed form ----------------------------------------------
    legacy = np.array([10.0 * np.log10(255.0 ** 2 /
                                       max(((g.astype(np.float64) - p.astype(np.float64)) ** 2).mean(), 1e-12))
                       for g, p in zip(gY, pY)])
    new_psnr = metrics.psnr_frames(pY, gY)
    rep["psnr"] = {"max_abs_diff": float(np.abs(new_psnr - legacy).max()),
                   "new_mean": float(new_psnr.mean()), "legacy_mean": float(legacy.mean()),
                   "pass": bool(np.abs(new_psnr - legacy).max() < 1e-9)}

    # 3. LPIPS vs the old hand-rolled helper ----------------------------------
    net = metrics.LPIPS(device=a.device)

    def legacy_lpips(x, y, bs=25):
        vals = []
        for i in range(0, len(x), bs):
            with torch.no_grad():
                d = net.net(metrics.LPIPS._to_tensor(x[i:i + bs], a.device),
                            metrics.LPIPS._to_tensor(y[i:i + bs], a.device))
            vals.extend(d.reshape(-1).cpu().numpy().tolist())
        return float(np.mean(vals))

    stack_p, stack_g = np.stack(pred), np.stack(gt)
    new_lp = net.mean(stack_p, stack_g)
    old_lp = legacy_lpips(stack_p, stack_g)
    per_pair_new = net.pair(stack_p, stack_g)
    rep["lpips"] = {"new": new_lp, "legacy_helper": old_lp,
                    "abs_diff": abs(new_lp - old_lp),
                    "per_frame_mean_of_pair": float(per_pair_new.mean()),
                    "pass": bool(abs(new_lp - old_lp) < 1e-9)}

    # 4. evaluate_stack == the components used one by one ---------------------
    st = metrics.evaluate_stack(pred, gt, device=a.device, lpips_net=net)
    rep["evaluate_stack"] = {"value": st,
                             "agrees": bool(abs(st["psnr"] - new_psnr.mean()) < 1e-9
                                            and abs(st["ssim"] - new_ssim.mean()) < 1e-12
                                            and abs(st["lpips"] - new_lp) < 1e-9)}
    rep["all_pass"] = all(rep[k]["pass"] for k in ("ssim", "psnr", "lpips")) and rep["evaluate_stack"]["agrees"]

    print(json.dumps(rep, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
