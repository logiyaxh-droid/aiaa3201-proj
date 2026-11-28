# -*- coding: utf-8 -*-
"""Run SUPIR v0F fp16 on the RealVSR subset.

Identical sampling configuration to the earlier qualification run (20 EDM
steps, fixed seed, fixed prompt, Wavelet colour fix, restoration_scale 1.0,
cfg 4.0) so the RealVSR rows are comparable with the VideoLQ/phone rows.

Only the I/O paths change: LQ comes from data/realvsr/LQ4x (128x256), is
upscaled x4 with LANCZOS, and the output must land on the GT grid 512x1024.

Usage:
  python run_supir_realvsr.py --limit 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor", "SUPIR")
sys.path.insert(0, VENDOR)

import torch
from PIL import Image

YAML = os.path.join(VENDOR, "options", "SUPIR_v0.yaml")
LQ4X = os.path.join(ROOT, "data", "realvsr", "LQ4x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--name", default="supir")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from SUPIR.util import PIL2Tensor, Tensor2PIL
    from supir_lean import create_supir_lean

    model = create_supir_lean(YAML, SUPIR_sign="F")
    model = model.to("cuda").eval()
    print("model ready", flush=True)

    prompt = ["high quality, detailed, sharp, natural texture"]
    out_root = os.path.join(ROOT, "outputs", "realvsr", args.name)
    clips = sorted(d for d in os.listdir(LQ4X)
                   if os.path.isdir(os.path.join(LQ4X, d)))
    if args.limit:
        clips = clips[:args.limit]

    t_all = time.time()
    n = 0
    for clip in clips:
        ids = sorted(os.path.splitext(f)[0]
                     for f in os.listdir(os.path.join(LQ4X, clip))
                     if f.lower().endswith(".png"))
        for fid in ids:
            dst = os.path.join(out_root, clip, fid + ".png")
            if os.path.exists(dst):
                continue
            img = Image.open(os.path.join(LQ4X, clip, fid + ".png")).convert("RGB")
            img = img.resize((img.width * 4, img.height * 4), Image.LANCZOS)
            x, h0, w0 = PIL2Tensor(img, min_size=1024)
            t0 = time.time()
            with torch.no_grad():
                out = model.batchify_sample(
                    x[None].half().cuda(), prompt,
                    num_steps=args.steps, restoration_scale=1.0,
                    s_churn=5.0, s_noise=1.01, cfg_scale=4.0,
                    seed=args.seed, control_scale=1.0,
                    color_fix_type="Wavelet",
                    use_linear_CFG=False, use_linear_control_scale=False,
                    cfg_scale_start=1.0, control_scale_start=0.0)
            pil = Tensor2PIL(out[0].float().cpu(), h0, w0)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            pil.save(dst)
            n += 1
            print("%s %s/%s %.1fs" % (args.name, clip, fid, time.time() - t0),
                  flush=True)
    print("DONE %d frames %.1f min" % (n, (time.time() - t_all) / 60), flush=True)


if __name__ == "__main__":
    main()
