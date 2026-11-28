#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rapid complementarity probe: cache EDVR predictions for the 40-frame screen.

Protocol (frozen before any StableSR output exists)
---------------------------------------------------
dev clips 016-019, frames 40-49, 10 frames each = 40 frames.
EDVR reads each clip's full 100-frame sequence and only the 40-49 outputs are
retained, so the five-frame windows use real neighbours and never clamp wrongly.
Quantisation matches the P0 pipeline: clamp -> *255 -> round -> uint8.

Usage:
    /tmp/srenv/bin/python scripts/cache_edvr_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
import hashlib

os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as _F

_mod = types.ModuleType("torchvision.transforms.functional_tensor")
for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _mod
torch.backends.cudnn.benchmark = False

BASE = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = os.path.join(BASE, "train_sample_sharp")
LR_ROOT = os.path.join(BASE, "train_sample_bicubic")
CK = ("/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
      "EDVR_L_x4_SR_REDS_official-9f5f5039.pth")
OUT = "/data/disk2/home/ghe270/p3/stablesr_fusion/probe/cache/edvr"
CLIPS = ["016", "017", "018", "019"]
FRAMES = list(range(40, 50))


def read_rgb(path):
    a = cv2.imread(path)
    if a is None:
        raise RuntimeError("unreadable frame: %s" % path)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


class EDVR:
    def __init__(self, device="cuda"):
        from basicsr.archs.edvr_arch import EDVR as Net
        ck = torch.load(CK, map_location="cpu", weights_only=False)["params"]
        net = Net(num_in_ch=3, num_out_ch=3, num_feat=128, num_frame=5,
                  deformable_groups=8, num_extract_block=5,
                  num_reconstruct_block=40, center_frame_idx=2,
                  hr_in=False, with_predeblur=False, with_tsa=True)
        net.load_state_dict(ck, strict=True)
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device

    def frame(self, lr, c):
        n = len(lr)
        win = [max(0, min(n - 1, c + d - 2)) for d in range(5)]
        x = torch.stack([torch.from_numpy(lr[w]).permute(2, 0, 1).float()
                         .div_(255.0) for w in win]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.net(x)
        return (y[0].clamp_(0, 1).mul_(255).round_().byte()
                .permute(1, 2, 0).cpu().numpy())


def main():
    os.makedirs(OUT, exist_ok=True)
    model = EDVR()
    manifest = {"protocol": {"clips": CLIPS, "frames": FRAMES,
                             "edvr_ckpt": CK,
                             "quantisation": "clamp->*255->round->uint8",
                             "window": "full clip, frame c +/- 2, clamp at clip ends",
                             "target": "centre frame"},
                "clips": {}}
    t_all = time.time()
    for clip in CLIPS:
        lr_dir = os.path.join(LR_ROOT, clip)
        gt_dir = os.path.join(GT_ROOT, clip)
        lr = [read_rgb(os.path.join(lr_dir, "%08d.png" % i)) for i in range(100)]
        out_dir = os.path.join(OUT, clip)
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        for f in FRAMES:
            t0 = time.time()
            pred = model.frame(lr, f)
            gt = read_rgb(os.path.join(gt_dir, "%08d.png" % f))
            if pred.shape != gt.shape:
                raise RuntimeError("clip %s frame %d: %s vs %s"
                                   % (clip, f, pred.shape, gt.shape))
            dst = os.path.join(out_dir, "%08d.png" % f)
            cv2.imwrite(dst, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
            rows.append({"frame": f, "path": dst,
                         "sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest(),
                         "elapsed_seconds": time.time() - t0})
            print("EDVR %s %03d %.2fs" % (clip, f, time.time() - t0), flush=True)
        manifest["clips"][clip] = rows
        with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)
    manifest["total_seconds"] = time.time() - t_all
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print("done", manifest["total_seconds"], flush=True)


if __name__ == "__main__":
    main()
