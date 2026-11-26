# -*- coding: utf-8 -*-
"""Produce the 4th demo panel: Ours (blend_sqrt) over a 100-frame window.

Same rule as the evaluated 9-frame blend: U1 = flip-disagreement of the frozen
reference model, normalised by its own q95, then sqrt(u) as the mixing weight.
Nothing is refitted here; the weight map is the quantity fine-tuning already used.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch

M2B = "/data/disk1/home/ghe270/p3/m2b_rbv_ft"
sys.path.insert(0, os.path.join(M2B, "scripts"))
sys.path.insert(0, M2B)

import warp as W                      # noqa: E402
from model_rbv_ft import UARBVSR       # noqa: E402

DATA = "/data/disk1/home/ghe270/p3/prior_qualification/data/reds_real"


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=["000", "016", "005"])
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--root", default="/tmp/p3_demo/out")
    ap.add_argument("--ref", default="/tmp/rbv_step0.pth")
    ap.add_argument("--chunk", type=int, default=9)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = UARBVSR("cuda")            # weights unused: only .ref is called
    for clip in args.clips:
        ids = sorted(int(os.path.splitext(f)[0])
                     for f in os.listdir(os.path.join(DATA, "LQ", clip))
                     if f.lower().endswith(".png"))[:args.frames]
        lq = np.stack([rd(os.path.join(DATA, "LQ", clip, "%08d.png" % i))
                       for i in ids])
        t = torch.from_numpy(lq).permute(0, 3, 1, 2).float().cuda() / 255.
        with torch.no_grad():
            b, bf = model.reference_forward(t)
            u = W.uncertainty_u1(b, bf)
        n = 0
        for k, fid in enumerate(ids):
            fn = "%08d.png" % fid
            y1 = rd(os.path.join(args.root, "b1", clip, fn))
            yr = rd(os.path.join(args.root, "rbv", clip, fn))
            if y1.shape != yr.shape:
                raise SystemExit("size mismatch %s/%s" % (clip, fn))
            h, w = y1.shape[:2]
            uk = u[k, 0].float().cpu().numpy()
            if uk.shape != (h, w):
                uk = cv2.resize(uk, (w, h), interpolation=cv2.INTER_LINEAR)
            a = np.sqrt(uk)[..., None].astype(np.float32)
            out = (1.0 - a) * yr + a * y1
            dst = os.path.join(args.root, "ours", clip, fn)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cv2.imwrite(dst, cv2.cvtColor(
                np.clip(out + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            n += 1
        print("[ours] %s %d frames" % (clip, n), flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
