# -*- coding: utf-8 -*-
"""Minimal SUPIR smoke test on the borrowed server (no LLaVA).

Follows official test.py load order: create -> half -> to(cuda).
Runs one 320x180 LR frame upscaled x4 to 1280x720 and records peak VRAM/time.

Usage:
  ./venv/bin/python run_supir_smoke.py --clip 000 --frame 40 --steps 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time

BASE = "/mnt/sda1/shidanhe/utrd_qual"
VENDOR = os.path.join(BASE, "vendor", "SUPIR")
WEIGHTS = os.path.join(BASE, "weights")
sys.path.insert(0, VENDOR)

import torch
from PIL import Image

YAML = os.path.join(VENDOR, "options", "SUPIR_v0.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="000")
    ap.add_argument("--frame", default="40")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-size", type=int, default=1024)
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL

    t0 = time.time()
    model = create_SUPIR_model(YAML, SUPIR_sign="F")
    if args.half:
        model = model.half()
    model = model.to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    print("model loaded %.1fs vram %.2f GiB"
          % (time.time() - t0, torch.cuda.memory_allocated() / 2**30), flush=True)

    lr_path = os.path.join(BASE, "input", "lr", args.clip,
                           "%08d.png" % int(args.frame))
    if not os.path.exists(lr_path):
        raise SystemExit("missing LR: %s" % lr_path)
    img = Image.open(lr_path).convert("RGB")
    img = img.resize((img.width * 4, img.height * 4), Image.LANCZOS)
    x, h0, w0 = PIL2Tensor(img, min_size=args.min_size)
    print("tensor", tuple(x.shape), "target", (h0, w0), flush=True)

    prompt = ["high quality, detailed, sharp, natural texture"]
    t1 = time.time()
    with torch.no_grad():
        out = model.batchify_sample(
            x[None].half().cuda(), prompt,
            num_steps=args.steps,
            restoration_scale=1.0,
            s_churn=5.0,
            s_noise=1.01,
            cfg_scale=4.0,
            seed=args.seed,
            control_scale=1.0,
            color_fix_type="Wavelet",
            use_linear_CFG=False,
            use_linear_control_scale=False,
            cfg_scale_start=1.0,
            control_scale_start=0.0,
        )
    print("sample %.1fs peak %.2f GiB"
          % (time.time() - t1, torch.cuda.max_memory_allocated() / 2**30), flush=True)

    pil = Tensor2PIL(out[0].float().cpu(), h0, w0)
    dst = args.out or os.path.join(BASE, "outputs", "supir_smoke",
                                   args.clip, "%08d.png" % int(args.frame))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pil.save(dst)
    print("saved", dst, pil.size, flush=True)


if __name__ == "__main__":
    main()
