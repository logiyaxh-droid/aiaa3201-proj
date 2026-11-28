# -*- coding: utf-8 -*-
"""Run EDVR (original / perceptual fine-tuned) on real-world frames.

Executed on the original server because it already has BasicSR, the DCD
extension and both checkpoints; the borrowed server has no basicsr.

Input : data/real/frames/wild01/<id>.png   720x1280 RGB
        EDVR is x4: it consumes the 180x320 bicubic downsample and reconstructs
        720x1280.  Five-frame windows use the real neighbouring frames.
Output: outputs/real/<name>/<id>.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/data/disk1/home/ghe270/p3/prior_qualification"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p3/utrd/scripts")
sys.path.insert(0, "/data/disk1/home/ghe270/p0")

import qa_common as Q  # noqa: E402
import model as UM       # noqa: E402

FRAMES = os.path.join(ROOT, "data/real/frames/wild01")
OUT = os.path.join(ROOT, "outputs/real")
FT_CK = ("/data/disk1/home/ghe270/p3/quality_metrics/fine_tune/runs/"
         "adv_lr3e5_d003_a001/step_000900.pth")


def build(finetuned):
    net = UM.FrozenEDVR("cuda")
    if finetuned:
        ck = torch.load(FT_CK, map_location="cpu", weights_only=False)
        net.net.load_state_dict(ck["model"], strict=True)
        net.net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="edvr")
    ap.add_argument("--finetuned", default="false")
    args = ap.parse_args()

    finetuned = args.finetuned.lower() in ("1", "true", "yes")
    net = build(finetuned)
    paths = sorted(glob.glob(os.path.join(FRAMES, "*.png")))
    if not paths:
        raise SystemExit("no frames in %s" % FRAMES)
    gts = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    lrs = []
    for g in gts:
        h, w = g.shape[:2]
        lrs.append(cv2.resize(g, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC))

    out_dir = os.path.join(OUT, args.name)
    os.makedirs(out_dir, exist_ok=True)
    n = len(lrs)
    t0 = time.time()
    for i, p in enumerate(paths):
        win = [lrs[max(0, min(n - 1, i + d - 2))] for d in range(5)]
        x = torch.stack([torch.from_numpy(a).permute(2, 0, 1).float().div(255)
                         for a in win])[None].cuda()
        y, _feat = net(x)
        out = y[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        h, w = gts[i].shape[:2]
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(p)),
                    cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        if (i + 1) % 10 == 0:
            print("%d/%d %.1fs" % (i + 1, n, time.time() - t0), flush=True)
    print("%s done %d frames in %.1fs" % (args.name, n, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
