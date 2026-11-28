# -*- coding: utf-8 -*-
"""Run official OSEDiff on the RealVSR subset, then score against GT.

Follows the official test_osediff.py chain exactly:
  LR -> LANCZOS x4 -> multiple of 8 -> [-1,1] tensor
     -> official OSEDiff_test model(lq, prompt) -> *0.5+0.5
     -> wavelet colour fix against the x4 upsampled input
     -> crop back to the GT grid

The only difference from the REDS run is the I/O paths and the final crop:
RealVSR GT is 512x1024, which is not a multiple of 8 (512 and 1024 are, but the
optimiser path pads anyway), so we pad the canvas, run the model, then crop.

Usage:
  python run_osediff_realvsr.py --name osediff --limit 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")
sys.path.insert(0, "/data/disk1/home/ghe270/p3/quality_metrics/scripts")

os.environ.setdefault(
    "TORCH_HOME", "/data/disk1/home/ghe270/p3/quality_metrics/weights/torch_home")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DATA = os.path.join(ROOT, "data", "realvsr")
LQ4X = os.path.join(DATA, "LQ4x")
GT = os.path.join(DATA, "GT")
VENDOR = os.path.join(ROOT, "vendor", "OSEDiff-main")
WEIGHTS = os.path.join(ROOT, "weights")
SD21 = os.path.join(WEIGHTS, "SD21BASE")
OSEDIFF_PKL = os.path.join(VENDOR, "preset", "models", "osediff.pkl")
DAPE = os.path.join(VENDOR, "preset", "models", "DAPE.pth")
RAM_CK = os.path.join(WEIGHTS, "RAM", "ram_swin_large_14m.pth")


def read_rgb(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def write_rgb(p, a):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


def load_model():
    sys.path.insert(0, VENDOR)
    from types import SimpleNamespace
    from osediff import OSEDiff_test

    args = SimpleNamespace(
        pretrained_model_name_or_path=SD21, osediff_path=OSEDIFF_PKL,
        mixed_precision="fp16", merge_and_unload_lora=False,
        vae_decoder_tiled_size=224, vae_encoder_tiled_size=1024,
        latent_tiled_size=96, latent_tiled_overlap=32,
    )
    model = OSEDiff_test(args)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="osediff")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ram", action="store_true")
    args = ap.parse_args()

    out_root = os.path.join(ROOT, "outputs", "realvsr", args.name)
    model = load_model()
    sys.path.insert(0, VENDOR)
    from my_utils.wavelet_color_fix import wavelet_color_fix

    ram_model, ram_tf = None, None
    if not args.no_ram and os.path.exists(RAM_CK):
        from ram.models.ram_lora import ram
        from ram import inference_ram as inference  # noqa: F401
        ram_model = ram(pretrained=RAM_CK, pretrained_condition=DAPE,
                        image_size=384, vit="swin_l")
        ram_model.eval().to("cuda")
        ram_tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        print("RAM loaded", flush=True)
    else:
        print("RAM disabled (prompt empty)", flush=True)

    clips = sorted(d for d in os.listdir(LQ4X)
                   if os.path.isdir(os.path.join(LQ4X, d)))
    if args.limit:
        clips = clips[:args.limit]
    t_all = time.time()
    n = 0
    for clip in clips:
        ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(LQ4X, clip))
                     if f.lower().endswith(".png"))
        for fid in ids:
            dst = os.path.join(out_root, clip, fid + ".png")
            if os.path.exists(dst):
                continue
            src = Image.open(os.path.join(LQ4X, clip, fid + ".png")).convert("RGB")
            width, height = src.size
            up = src.resize((width * 4, height * 4), Image.LANCZOS)
            # official pipeline crops to a multiple of 8; pad instead so the
            # output grid is exactly 4x the LQ grid, then crop at the end.
            pw = (8 - up.width % 8) % 8
            ph = (8 - up.height % 8) % 8
            canvas = Image.new("RGB", (up.width + pw, up.height + ph))
            canvas.paste(up, (0, 0))
            lq = transforms.ToTensor()(canvas).unsqueeze(0).cuda() * 2 - 1

            prompt = ""
            if ram_model is not None:
                from ram import inference_ram as inference
                cap = inference(ram_tf(transforms.ToTensor()(canvas).unsqueeze(0).cuda()),
                                ram_model)[0]
                prompt = f"{cap}, "

            torch.manual_seed(args.seed)
            with torch.no_grad():
                out = model(lq.half(), prompt=prompt)
            pil = transforms.ToPILImage()(out[0].float().cpu() * 0.5 + 0.5)
            pil = wavelet_color_fix(target=pil, source=canvas)
            pil = pil.crop((0, 0, up.width, up.height))
            Q_out = np.asarray(pil)
            write_rgb(dst, Q_out)
            n += 1
            print("%s %s/%s %.1fs" % (args.name, clip, fid,
                                      time.time() - t_all), flush=True)

    print("DONE %d frames in %.1fs" % (n, time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
