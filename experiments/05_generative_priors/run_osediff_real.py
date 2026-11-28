# -*- coding: utf-8 -*-
"""Run OSEDiff on the real-world frames.

Protocol: the 720x1280 frame is downsampled x4 to 180x320 (bicubic), then
OSEDiff upsamples back to 720x1280.  This isolates "how well does the model
restore what x4 downsampling destroyed" and matches the EDVR protocol exactly.

Output: outputs/real/<name>/<id>.png  720x1280 RGB
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

BASE = "/mnt/sda1/shidanhe/utrd_qual"
VENDOR = os.path.join(BASE, "vendor", "OSEDiff")
SD21 = os.path.join(BASE, "weights_sd21")
FRAMES = os.path.join(BASE, "data/wild/frames/wild01")
OSEDIFF_PKL = os.path.join(VENDOR, "preset/models/osediff.pkl")


def load_model():
    sys.path.insert(0, VENDOR)
    from osediff import OSEDiff_test
    args = SimpleNamespace(
        pretrained_model_name_or_path=SD21, osediff_path=OSEDIFF_PKL,
        mixed_precision="fp16", merge_and_unload_lora=False,
        vae_decoder_tiled_size=224, vae_encoder_tiled_size=1024,
        latent_tiled_size=96, latent_tiled_overlap=32)
    m = OSEDiff_test(args)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="osediff")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = load_model()
    sys.path.insert(0, VENDOR)
    from my_utils.wavelet_color_fix import wavelet_color_fix

    paths = sorted(glob.glob(os.path.join(FRAMES, "*.png")))
    out_dir = os.path.join(BASE, "outputs/real", args.name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for k, p in enumerate(paths):
        src = Image.open(p).convert("RGB")
        w, h = src.size
        lr = src.resize((w // 4, h // 4), Image.BICUBIC)
        up = lr.resize((w, h), Image.LANCZOS)
        x = transforms.ToTensor()(up).unsqueeze(0).cuda() * 2 - 1
        torch.manual_seed(args.seed)
        with torch.no_grad():
            out = model(x.half(), prompt="")
        pil = transforms.ToPILImage()(out[0].float().cpu() * 0.5 + 0.5)
        pil = wavelet_color_fix(target=pil, source=up)
        if pil.size != (w, h):
            pil = pil.resize((w, h), Image.LANCZOS)
        pil.save(os.path.join(out_dir, os.path.basename(p)))
        if (k + 1) % 5 == 0:
            print("%d/%d %.1fs vram %.2fGiB"
                  % (k + 1, len(paths), time.time() - t0,
                     torch.cuda.max_memory_allocated() / 2**30), flush=True)
    print("done %d frames %.1f min" % (len(paths), (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
