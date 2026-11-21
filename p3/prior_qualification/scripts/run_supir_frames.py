# -*- coding: utf-8 -*-
"""Run SUPIR over the 30 frozen qualification frames.

Same fixed configuration as the smoke test (v0F fp16, 20 EDM steps, fixed seed,
fixed prompt).  Each LR frame is upscaled x4 to 1280x720, SUPIR runs, output is
saved as lossless PNG for scoring on the original server.

Usage:
  ./venv/bin/python run_supir_frames.py [--steps 20]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

BASE = "/mnt/sda1/shidanhe/utrd_qual"
VENDOR = os.path.join(BASE, "vendor", "SUPIR")
sys.path.insert(0, VENDOR)

import torch
from PIL import Image

YAML = os.path.join(VENDOR, "options", "SUPIR_v0.yaml")
CLIPS = ("000", "002", "009")
FRAMES = range(40, 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="supir")
    args = ap.parse_args()

    from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL

    model = create_SUPIR_model(YAML, SUPIR_sign="F")
    model = model.half().to("cuda").eval()
    print("model ready", flush=True)

    prompt = ["high quality, detailed, sharp, natural texture"]
    out_root = os.path.join(BASE, "outputs", args.out)
    t_all = time.time()
    for clip in CLIPS:
        for fid in FRAMES:
            dst = os.path.join(out_root, clip, "%08d.png" % fid)
            if os.path.exists(dst):
                continue
            src = os.path.join(BASE, "input", "lr", clip, "%08d.png" % fid)
            img = Image.open(src).convert("RGB")
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
            print("%s/%08d %.1fs peak %.2f GiB"
                  % (clip, fid, time.time() - t0,
                     torch.cuda.max_memory_allocated() / 2**30), flush=True)
    print("done %.1f min" % ((time.time() - t_all) / 60), flush=True)


if __name__ == "__main__":
    main()
