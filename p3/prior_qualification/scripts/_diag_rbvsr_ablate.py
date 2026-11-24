# -*- coding: utf-8 -*-
"""Attribute the gain: which fix contributes how much, and where the uniform
+4/255 offset comes from."""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _diag_rbvsr_bugs as D  # noqa: E402

ROOT = D.ROOT
DATA = D.DATA


def main():
    lq_dir = os.path.join(DATA, "LQ4x", D.CLIP)
    gt_dir = os.path.join(DATA, "GT", D.CLIP)
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(lq_dir)
                 if f.lower().endswith(".png"))[:5]
    lq = np.stack([D.read_rgb(os.path.join(lq_dir, i + ".png")) for i in ids])
    gt = np.stack([D.read_rgb(os.path.join(gt_dir, i + ".png")) for i in ids])
    g = gt.astype(np.float64)
    x = torch.from_numpy(lq).permute(0, 3, 1, 2).float().div_(255.0).cuda()

    combos = {"lrelu+warp": ["lrelu", "warp"], "lrelu+pad": ["lrelu", "pad"]}
    out = {}
    for name, names in combos.items():
        mod = D.build_module(name.replace("+", "_"), names)
        net, m, u = mod.load_rbvsr(D.CKPT, device="cuda")
        with torch.no_grad():
            y = net(x).clamp(0, 1).mul(255.0).permute(0, 2, 3, 1)
            y = y.round().byte().cpu().numpy().astype(np.float64)
        per = [D.psnr(y[i], g[i]) for i in range(len(ids))]
        out[name] = {"psnr": float(np.mean(per)),
                     "bias": [round(float(v), 2)
                              for v in (y - g).mean(axis=(0, 1, 2))]}
        print("%-12s PSNR %.4f  bias %s" % (name, out[name]["psnr"],
                                            out[name]["bias"]), flush=True)
        del net
        torch.cuda.empty_cache()

    # how much DC the cleaning module injects: that is what the base image
    # carries, since base = bilinear(cleaned)
    mod = D.build_module("full", ["lrelu", "warp", "pad"])
    net, _, _ = mod.load_rbvsr(D.CKPT, device="cuda")
    with torch.no_grad():
        res = net.image_cleaning(x)
        mean_res = res.mean().item() * 255.0
        abs_mean = res.abs().mean().item() * 255.0
        cleaned = (x + res).clamp(0, 1)
        print("cleaning residual: signed mean %+.3f/255  abs mean %.3f/255"
              % (mean_res, abs_mean))
        print("input mean %.3f  cleaned mean %.3f"
              % (float(x.mean()) * 255, float(cleaned.mean()) * 255))
        out["cleaning_residual_dc"] = round(mean_res, 3)
        print("early-stop: abs mean %.4f vs thres %.4f -> runs 1 pass"
              % (abs_mean / 255.0, net.dynamic_refine_thres))

    dst = os.path.join(ROOT, "results", "rbvsr_diag", "ablate.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
