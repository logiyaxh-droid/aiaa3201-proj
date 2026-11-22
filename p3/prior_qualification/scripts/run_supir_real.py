# -*- coding: utf-8 -*-
"""Run SUPIR v0F on the real-world frames (no LLaVA, fixed prompt).

Protocol: the 720x1280 frame is downsampled x4 to 180x320, then SUPIR restores
it back to 720x1280.  Identical input path to EDVR and OSEDiff so the three
methods are comparable.

Output: outputs/real/supir/<id>.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

BASE = "/mnt/sda1/shidanhe/utrd_qual"
VENDOR = os.path.join(BASE, "vendor", "SUPIR")
W = os.path.join(BASE, "weights_supir")
FRAMES = os.path.join(BASE, "data/wild/frames/wild01")
sys.path.insert(0, VENDOR)

import torch
from PIL import Image

YAML = os.path.join(VENDOR, "options", "SUPIR_v0.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="supir")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL

    model = create_SUPIR_model(YAML, SUPIR_sign="F")
    model = model.half().to("cuda").eval()
    print("model ready", flush=True)

    prompt = ["high quality, detailed, sharp, natural texture"]
    paths = sorted(glob.glob(os.path.join(FRAMES, "*.png")))
    out_dir = os.path.join(BASE, "outputs/real", args.name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for k, p in enumerate(paths):
        src = Image.open(p).convert("RGB")
        w, h = src.size
        lr = src.resize((w // 4, h // 4), Image.BICUBIC)
        up = lr.resize((w, h), Image.LANCZOS)
        x, h0, w0 = PIL2Tensor(up, min_size=1024)
        t1 = time.time()
        with torch.no_grad():
            out = model.batchify_sample(
                x[None].half().cuda(), prompt, num_steps=args.steps,
                restoration_scale=1.0, s_churn=5.0, s_noise=1.01,
                cfg_scale=4.0, seed=args.seed, control_scale=1.0,
                color_fix_type="Wavelet", use_linear_CFG=False,
                use_linear_control_scale=False, cfg_scale_start=1.0,
                control_scale_start=0.0)
        pil = Tensor2PIL(out[0].float().cpu(), h0, w0)
        if pil.size != (w, h):
            pil = pil.resize((w, h), Image.LANCZOS)
        pil.save(os.path.join(out_dir, os.path.basename(p)))
        if (k + 1) % 5 == 0:
            print("%d/%d %.1fs/frame peak %.2fGiB"
                  % (k + 1, len(paths), (time.time() - t0) / (k + 1),
                     torch.cuda.max_memory_allocated() / 2**30), flush=True)
    print("done %d frames %.1f min" % (len(paths), (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
