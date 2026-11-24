# -*- coding: utf-8 -*-
"""Final confirmation for the rbvsr_net.py fixes.

Reports, on RealVSR clip 016, using the same luma PSNR as the rest of the
study (EVALKIT metrics.psnr_mean):
  * all three fixes, RGB input
  * all three fixes, BGR input   (channel-order sanity check)
  * bicubic baseline recomputed here, so the comparison is on one footing
"""
from __future__ import annotations

import json
import os
import sys
import types

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")
SRC = os.path.join(ROOT, "scripts", "rbvsr_net.py")
CKPT = os.path.join(ROOT, "weights_rbvsr", "RealBasicVSR_x4.pth")
DATA = os.environ.get("RVQ_DATA") or os.path.join(ROOT, "data", "realvsr")
CLIP = "016"

FIXED = [
    ("            x = self.basicvsr.upsample1(x)\n"
     "            x = self.basicvsr.upsample2(x)\n",
     "            x = F.leaky_relu(self.basicvsr.upsample1(x),"
     " negative_slope=0.1, inplace=True)\n"
     "            x = F.leaky_relu(self.basicvsr.upsample2(x),"
     " negative_slope=0.1, inplace=True)\n"),
    ("    grid = grid + flow.flip(1)\n", "    grid = grid + flow\n"),
    ("def _warp(x, flow):", "def _warp(x, flow, padding_mode=\"zeros\"):"),
    ('mode="bilinear", padding_mode="border")',
     'mode="bilinear", padding_mode=padding_mode)'),
    ("                ref[level], _warp(supp[level], up), up], 1)) + up",
     "                ref[level], _warp(supp[level], up, padding_mode=\"border\"),"
     " up], 1)) + up"),
]


def build():
    src = open(SRC, encoding="utf-8").read()
    for old, new in FIXED:
        if old not in src:
            raise SystemExit("anchor missing: %r" % old[:70])
        src = src.replace(old, new, 1)
    mod = types.ModuleType("rbvsr_fixed")
    mod.__file__ = SRC
    exec(compile(src, SRC, "exec"), mod.__dict__)
    return mod


def read_rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def main():
    from evalkit import metrics as EM

    lq_dir = os.path.join(DATA, "LQ4x", CLIP)
    gt_dir = os.path.join(DATA, "GT", CLIP)
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(lq_dir)
                 if f.lower().endswith(".png"))[:5]
    lq = np.stack([read_rgb(os.path.join(lq_dir, i + ".png")) for i in ids])
    gt = np.stack([read_rgb(os.path.join(gt_dir, i + ".png")) for i in ids])

    def luma_psnr(pred):
        pY = np.stack([EM.luma(a) for a in pred])
        gY = np.stack([EM.luma(b) for b in gt])
        return float(EM.psnr_mean(pY, gY))

    bic = np.stack([cv2.resize(a, (a.shape[1] * 4, a.shape[0] * 4),
                               interpolation=cv2.INTER_CUBIC) for a in lq])

    mod = build()
    net, m, u = mod.load_rbvsr(CKPT, device="cuda")
    out = {"ids": ids, "missing": len(m), "unexpected": len(u)}
    for order in ("rgb", "bgr"):
        src = lq if order == "rgb" else lq[..., ::-1]
        x = torch.from_numpy(np.ascontiguousarray(src)).permute(0, 3, 1, 2)
        x = x.float().div_(255.0).cuda()
        with torch.no_grad():
            y = net(x).clamp(0, 1).mul(255.0).permute(0, 2, 3, 1)
            y = y.round().byte().cpu().numpy()
        out["rbvsr_" + order] = luma_psnr(y)
        print("rbvsr (%s input)  luma PSNR %.4f" % (order, out["rbvsr_" + order]),
              flush=True)

    # bias of the RGB-order prediction, the channel-order check the report asks for
    src = lq
    x = torch.from_numpy(src).permute(0, 3, 1, 2).float().div_(255.0).cuda()
    with torch.no_grad():
        y = net(x).clamp(0, 1).mul(255.0).permute(0, 2, 3, 1)
        y = y.round().byte().cpu().numpy().astype(np.float64)
    out["bias_rgb_order"] = [round(float(v), 2) for v in
                             (y - gt.astype(np.float64)).mean(axis=(0, 1, 2))]

    out["bicubic"] = luma_psnr(bic)
    print("bicubic baseline   luma PSNR %.4f" % out["bicubic"])
    print("bias(RGB) fixed =", out["bias_rgb_order"])

    dst = os.path.join(ROOT, "results", "rbvsr_diag", "final_check.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
