# -*- coding: utf-8 -*-
"""Run official OSEDiff on the frozen qualification frames.

Follows the official test_osediff.py chain exactly:
  LR -> LANCZOS x4 -> multiple of 8 -> [-1,1] tensor
     -> official OSEDiff_test model(lq, prompt) -> *0.5+0.5
     -> wavelet colour fix against the x4 upsampled input
     -> 1280x720 RGB PNG

Each frame is independent (image model, no temporal module).  Seeds are fixed.

Usage:
  python run_osediff.py --name osediff_smoke --clips 000 --frames 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import qa_common as Q

VENDOR = os.path.join(ROOT, "vendor", "OSEDiff-main")
WEIGHTS = os.path.join(ROOT, "weights")
SD21 = os.path.join(WEIGHTS, "SD21BASE")
OSEDIFF_PKL = os.path.join(VENDOR, "preset", "models", "osediff.pkl")
DAPE = os.path.join(VENDOR, "preset", "models", "DAPE.pth")
RAM_CK = os.path.join(WEIGHTS, "RAM", "ram_swin_large_14m.pth")


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
    ap.add_argument("--clips", nargs="*", default=list(Q.DEV_CLIPS))
    ap.add_argument("--frames", type=int, default=10,
                    help="frames per clip starting at frame 40")
    ap.add_argument("--no-ram", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = os.path.join(ROOT, "outputs", args.name)
    os.makedirs(out_root, exist_ok=True)

    model = load_model()
    sys.path.insert(0, VENDOR)
    from my_utils.wavelet_color_fix import wavelet_color_fix

    ram_model = None
    if not args.no_ram:
        if not os.path.exists(RAM_CK):
            raise SystemExit("RAM weight missing: %s" % RAM_CK)
        from ram.models.ram_lora import ram
        from ram import inference_ram as inference
        ram_model = ram(pretrained=RAM_CK, pretrained_condition=DAPE,
                        image_size=384, vit="swin_l")
        ram_model.eval().to("cuda")
        ram_tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        print("RAM loaded", flush=True)

    t_all = time.time()
    for clip in args.clips:
        ids = Q.dev_ids(clip)[:args.frames]
        for fid in ids:
            dst = os.path.join(out_root, clip, "%08d.png" % fid)
            if os.path.exists(dst):
                continue
            src = Image.open(os.path.join(Q.LR_ROOT, clip, "%08d.png" % fid)).convert("RGB")
            width, height = src.size
            # official: LANCZOS x4, then crop to a multiple of 8
            up = src.resize((width * 4, height * 4), Image.LANCZOS)
            up = up.resize((up.width - up.width % 8, up.height - up.height % 8),
                           Image.LANCZOS)
            lq = transforms.ToTensor()(up).unsqueeze(0).cuda() * 2 - 1

            prompt = ""
            if ram_model is not None:
                from ram import inference_ram as inference
                cap = inference(ram_tf(transforms.ToTensor()(up).unsqueeze(0).cuda()),
                                ram_model)[0]
                prompt = f"{cap}, "

            torch.manual_seed(args.seed)
            t0 = time.time()
            with torch.no_grad():
                out = model(lq.half(), prompt=prompt)
            pil = transforms.ToPILImage()(out[0].float().cpu() * 0.5 + 0.5)
            pil = wavelet_color_fix(target=pil, source=up)
            if pil.size != (1280, 720):
                pil = pil.resize((1280, 720), Image.LANCZOS)
            Q.write_rgb(dst, np.asarray(pil))
            print("%s %s/%08d %.1fs vram %.2f GiB"
                  % (args.name, clip, fid, time.time() - t0,
                     torch.cuda.max_memory_allocated() / 2**30), flush=True)
    print("done %.1f min" % ((time.time() - t_all) / 60), flush=True)


if __name__ == "__main__":
    main()
