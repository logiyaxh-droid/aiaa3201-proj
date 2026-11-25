# -*- coding: utf-8 -*-
"""M2-B inference on REDS-Real c9 (whole-clip, no tiling needed in pixel space).

Why this is fast
----------------
RealBasicVSR is fully convolutional and propagates features through time, so a
whole 9-frame clip is ONE forward pass -- no latent tiling, no VAE round trip.
Peak VRAM for a 320x180 -> 1280x720 clip is ~1.5 GB and 270 frames take under
two minutes, versus the 8x-VAE pipeline's minutes per frame.

Usage:
  python eval_rbv_ft.py --run runs/B2_uncertain/last.pth --name b2 \
      --clips 000 002 009
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
M2B = os.path.dirname(HERE)
sys.path.insert(0, M2B)

from model_rbv_ft import UARBVSR          # noqa: E402
from train_rbv_ft import load_ckpt        # noqa: E402

DATA = os.environ.get("RBVSR_DATA") or \
    "/data/disk1/home/ghe270/p3/prior_qualification/data/reds_real_c9"
OUT_ROOT = os.path.join(M2B, "outputs")


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=9)
    args = ap.parse_args()

    clips = args.clips or sorted(os.listdir(os.path.join(DATA, "GT")))
    out_dir = os.path.join(OUT_ROOT, args.name)
    t0 = time.time()
    torch.manual_seed(0)
    model = UARBVSR("cuda")
    step, _ = load_ckpt(args.run, model)
    model.net.eval()
    print("[load] %s step=%d" % (args.run, step), flush=True)

    total = 0
    with torch.no_grad():
        for clip in clips:
            lq_dir = os.path.join(DATA, "LQ", clip)
            ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(lq_dir)
                         if f.lower().endswith(".png"))
            if args.limit:
                ids = ids[:args.limit]
            lq = np.stack([rd(os.path.join(lq_dir, "%08d.png" % i))
                           for i in ids])
            t = torch.from_numpy(lq).permute(0, 3, 1, 2).float().cuda() / 255.
            y = model.infer_sequence(t, chunk=args.chunk)
            for k, fid in enumerate(ids):
                dst = os.path.join(out_dir, clip, "%08d.png" % fid)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                arr = (y[k].clamp(0, 1).float().cpu().numpy()
                       .transpose(1, 2, 0) * 255.0 + 0.5)
                cv2.imwrite(dst, cv2.cvtColor(arr.astype(np.uint8),
                                              cv2.COLOR_RGB2BGR))
            total += len(ids)
            print("[%s] %d frames  %.1f s" % (clip, len(ids),
                                              time.time() - t0), flush=True)

    meta = {"run": args.run, "step": step, "clips": clips, "frames": total,
            "dir": out_dir, "elapsed_sec": time.time() - t0}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(OUT_ROOT, args.name + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("[done] %d frames in %.1f s -> %s"
          % (total, time.time() - t0, out_dir), flush=True)


if __name__ == "__main__":
    main()
