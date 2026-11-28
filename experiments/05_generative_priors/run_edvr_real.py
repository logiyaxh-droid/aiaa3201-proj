# -*- coding: utf-8 -*-
"""Run EDVR (original or perceptual fine-tuned) on the real-world frames.

Input : data/wild/frames/<clip>/<id>.png  720x1280 RGB
        EDVR is x4, so it consumes the 180x320 downsample of that frame and
        reconstructs back to 720x1280.  This matches how the REDS protocol
        feeds EDVR (LR = GT/4), except here the LR is derived from real
        footage rather than from a synthetic bicubic kernel.
Output: outputs/real/<name>/<id>.png      720x1280 RGB

Usage:
  python run_edvr_real.py --name edvr --finetuned false
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import types
import time

import cv2
import numpy as np
import torch
import torchvision

mod = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F  # noqa: E402

for n in dir(_F):
    setattr(mod, n, getattr(_F, n))
sys.modules["torchvision.transforms.functional_tensor"] = mod

BASE = "/mnt/sda1/shidanhe/utrd_qual"
FRAMES = os.path.join(BASE, "data/wild/frames/wild01")
WEIGHTS = os.path.join(BASE, "weights")
EDVR_CK = os.path.join(WEIGHTS, "EDVR_L_x4_SR_REDS_official-9f5f5039.pth")
FT_CK = os.path.join(WEIGHTS, "step_000900.pth")


def build(finetuned):
    from basicsr.archs.edvr_arch import EDVR
    net = EDVR(num_in_ch=3, num_out_ch=3, num_feat=128, num_frame=5,
               deformable_groups=8, num_extract_block=5,
               num_reconstruct_block=40, center_frame_idx=2,
               hr_in=False, with_predeblur=False, with_tsa=True)
    ck = torch.load(FT_CK if finetuned else EDVR_CK, map_location="cpu",
                    weights_only=False)
    state = ck["model"] if "model" in ck else ck["params"]
    net.load_state_dict(state, strict=True)
    return net.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="edvr")
    ap.add_argument("--finetuned", default="false")
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()

    finetuned = args.finetuned.lower() in ("1", "true", "yes")
    net = build(finetuned)
    paths = sorted(glob.glob(os.path.join(FRAMES, "*.png")))
    if not paths:
        raise SystemExit("no frames in %s" % FRAMES)
    out_dir = os.path.join(BASE, "outputs/real", args.name)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    # Pre-compute LR for every frame so each output can use its real neighbours.
    all_gt = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    lrs = []
    for gt in all_gt:
        h, w = gt.shape[:2]
        lrs.append(cv2.resize(gt, (w // args.scale, h // args.scale),
                              interpolation=cv2.INTER_CUBIC))
    n = len(lrs)
    for i, p in enumerate(paths):
        lr_list = [lrs[max(0, min(n - 1, i + d - 2))] for d in range(5)]
        x = torch.stack([torch.from_numpy(a).permute(2, 0, 1).float().div(255)
                         for a in lr_list])[None].cuda()
        with torch.no_grad():
            y = net(x)[0]
        out = y.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        h, w = all_gt[i].shape[:2]
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(p)),
                    cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print("%s done %d frames in %.1fs" % (args.name, len(paths), time.time() - t0),
          flush=True)


if __name__ == "__main__":
    main()
