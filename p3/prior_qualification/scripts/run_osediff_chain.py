# -*- coding: utf-8 -*-
"""RealBasicVSR -> OSEDiff zero-shot chain on REDS-Real c9.

Difference from run_osediff.py: the x4 stage is RealBasicVSR's own output
instead of LANCZOS.  Everything else follows the official test chain, including
the colour fix -- but `wavelet_color_fix(source=...)` is pointed at the
RealBasicVSR output, not at the LANCZOS upscale, so the fix does not undo what
RealBasicVSR restored.

Input : outputs/reds_real_c9/<rbvsr_name>/<clip>/<frame>.png
Output: outputs/reds_real_c9/<name>/<clip>/<frame>.png

Usage:
  python run_osediff_chain.py --name rbv_osediff --src gen_thres255_generator --limit 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")

os.environ.setdefault(
    "TORCH_HOME", "/data/disk1/home/ghe270/p3/quality_metrics/weights/torch_home")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# OSEDiff needs the diffusers 0.25 wheel kept in site_old/.
_OLD = os.path.join(ROOT, "site_old")
if os.path.isdir(os.path.join(_OLD, "diffusers")):
    sys.path.insert(0, _OLD)

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DATA = os.environ.get("RBVSR_DATA") or os.path.join(ROOT, "data", "reds_real_c9")
GT = os.path.join(DATA, "GT")
OUT = os.path.join(ROOT, "outputs", "reds_real_c9")
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
    ap.add_argument("--name", default="rbv_osediff")
    ap.add_argument("--src", required=True,
                    help="subdirectory under outputs/reds_real_c9 holding the x4 frames")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ram", action="store_true")
    args = ap.parse_args()

    src_root = os.path.join(OUT, args.src)
    if not os.path.isdir(src_root):
        raise SystemExit("missing x4 input dir %s" % src_root)
    dst_root = os.path.join(OUT, args.name)

    model = load_model()
    sys.path.insert(0, VENDOR)
    from my_utils.wavelet_color_fix import wavelet_color_fix

    ram_model, ram_tf = None, None
    if not args.no_ram and os.path.exists(RAM_CK):
        from ram.models.ram_lora import ram
        ram_model = ram(pretrained=RAM_CK, pretrained_condition=DAPE,
                        image_size=384, vit="swin_l")
        ram_model.eval().to("cuda")
        ram_tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        print("RAM loaded", flush=True)
    else:
        print("RAM disabled (empty prompt)", flush=True)

    clips = sorted(d for d in os.listdir(src_root)
                   if os.path.isdir(os.path.join(src_root, d)))
    if args.limit:
        clips = clips[:args.limit]
    t_all, n = time.time(), 0
    for clip in clips:
        ids = sorted(os.path.splitext(f)[0] for f in
                     os.listdir(os.path.join(src_root, clip))
                     if f.lower().endswith(".png"))
        # the GT grid is authoritative for the output size
        gt0 = read_rgb(os.path.join(GT, clip, ids[0] + ".png"))
        gth, gtw = gt0.shape[:2]
        for fid in ids:
            dst = os.path.join(dst_root, clip, fid + ".png")
            if os.path.exists(dst):
                continue
            src = Image.open(os.path.join(src_root, clip, fid + ".png")).convert("RGB")
            # RealBasicVSR output is already on the GT grid; only pad to a
            # multiple of 8 so the VAE sees the size the official chain expects.
            up = src if src.size == (gtw, gth) else src.resize((gtw, gth), Image.LANCZOS)
            pw = (8 - up.width % 8) % 8
            ph = (8 - up.height % 8) % 8
            if pw or ph:
                canvas = Image.new("RGB", (up.width + pw, up.height + ph))
                canvas.paste(up, (0, 0))
            else:
                canvas = up
            lq = transforms.ToTensor()(canvas).unsqueeze(0).cuda() * 2 - 1

            prompt = ""
            if ram_model is not None:
                from ram import inference_ram as inference
                cap = inference(ram_tf(transforms.ToTensor()(canvas)
                                      .unsqueeze(0).cuda()), ram_model)[0]
                prompt = f"{cap}, "

            torch.manual_seed(args.seed)
            with torch.no_grad():
                out = model(lq.half(), prompt=prompt)
            pil = transforms.ToPILImage()(out[0].float().cpu() * 0.5 + 0.5)
            # colour fix anchored on the RealBasicVSR frame, not on LANCZOS
            pil = wavelet_color_fix(target=pil, source=canvas)
            pil = pil.crop((0, 0, up.width, up.height))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cv2.imwrite(dst, cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
            n += 1
            print("%s %s/%s %.1fs" % (args.name, clip, fid,
                                      time.time() - t_all), flush=True)
    print("DONE %d frames in %.1fs" % (n, time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
