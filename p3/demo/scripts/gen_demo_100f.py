# -*- coding: utf-8 -*-
"""Generate 100-frame 4-way material on REDS-Real for the M4 demo video.

Four panels per frame, all on the SAME 100-frame window of the SAME clip:
  EDVR-L (x4, official REDS weights, 5-frame sliding windows)
  RealBasicVSR (step-0 checkpoint == the published 6.29M model)
  OSEDiff (official chain, per-frame image model)
  Ours (blend_sqrt) = sqrt(U1)-weighted mix of RealBasicVSR and B1

Everything is written as 720p PNG under one scratch root so the video builder
reads a single directory tree and frame alignment can never drift.

Usage:
  python gen_demo_100f.py --clips 000 016 005 --frames 100 --root /tmp/p3_demo/out --which rbv,b1,edvr
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

M2B = "/data/disk1/home/ghe270/p3/m2b_rbv_ft"
PRIOR = "/data/disk1/home/ghe270/p3/prior_qualification"
sys.path.insert(0, os.path.join(M2B, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p3/utrd/scripts")
sys.path.insert(0, "/data/disk1/home/ghe270/p0")

DATA = os.path.join(PRIOR, "data", "reds_real")


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def wr(p, a):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


def frame_ids(clip, n):
    d = os.path.join(DATA, "LQ", clip)
    ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(d)
                 if f.lower().endswith(".png"))
    return ids[:n]


def run_rbv(root, clips, n, run):
    """RealBasicVSR step-0 / fine-tuned, whole-clip chunked forward."""
    from model_rbv_ft import UARBVSR
    from train_rbv_ft import load_ckpt

    torch.manual_seed(0)
    model = UARBVSR("cuda")
    step, _ = load_ckpt(run, model)
    model.net.eval()
    print("[rbv] %s step=%d" % (run, step), flush=True)
    tag = "rbv" if "step0" in run else "b1"
    t0 = time.time()
    with torch.no_grad():
        for clip in clips:
            ids = frame_ids(clip, n)
            lq = np.stack([rd(os.path.join(DATA, "LQ", clip, "%08d.png" % i))
                           for i in ids])
            t = torch.from_numpy(lq).permute(0, 3, 1, 2).float().cuda() / 255.
            y = model.infer_sequence(t, chunk=9)
            for k, fid in enumerate(ids):
                arr = (y[k].clamp(0, 1).float().cpu().numpy()
                       .transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
                wr(os.path.join(root, tag, clip, "%08d.png" % fid), arr)
            print("[%s] %s %d frames %.1fs" % (tag, clip, len(ids),
                                               time.time() - t0), flush=True)


def run_edvr(root, clips, n):
    """Official EDVR-L: 5-frame sliding windows, centre frame is the output."""
    from model import FrozenEDVR

    net = FrozenEDVR("cuda")
    print("[edvr] loaded", flush=True)
    t0 = time.time()
    for clip in clips:
        ids = frame_ids(clip, n)
        lq = [rd(os.path.join(DATA, "LQ", clip, "%08d.png" % i)) for i in ids]
        arr = np.stack(lq).astype(np.float32) / 255.
        T = len(arr)
        for k in range(T):
            idx = [min(max(k + d, 0), T - 1) for d in (-2, -1, 0, 1, 2)]
            win = torch.from_numpy(arr[idx]).permute(0, 3, 1, 2)[None].cuda()
            out, _ = net(win)
            img = (out[0].clamp(0, 1).float().cpu().numpy()
                   .transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
            wr(os.path.join(root, "edvr", clip, "%08d.png" % ids[k]), img)
        print("[edvr] %s %d frames %.1fs" % (clip, T, time.time() - t0), flush=True)


def run_osediff(root, clips, n, seed=42):
    """Official OSEDiff chain, one independent pass per frame."""
    VENDOR = os.path.join(PRIOR, "vendor", "OSEDiff-main")
    OLD = os.path.join(PRIOR, "site_old")
    if os.path.isdir(os.path.join(OLD, "diffusers")):
        sys.path.insert(0, OLD)
    sys.path.insert(0, VENDOR)
    from types import SimpleNamespace
    from osediff import OSEDiff_test
    from my_utils.wavelet_color_fix import wavelet_color_fix
    from PIL import Image
    from torchvision import transforms

    args = SimpleNamespace(
        pretrained_model_name_or_path=os.path.join(PRIOR, "weights", "SD21BASE"),
        osediff_path=os.path.join(VENDOR, "preset", "models", "osediff.pkl"),
        mixed_precision="fp16", merge_and_unload_lora=False,
        vae_decoder_tiled_size=224, vae_encoder_tiled_size=1024,
        latent_tiled_size=96, latent_tiled_overlap=32)
    model = OSEDiff_test(args)
    model.eval()
    print("[osediff] loaded", flush=True)
    t0 = time.time()
    for clip in clips:
        ids = frame_ids(clip, n)
        for k, fid in enumerate(ids):
            dst = os.path.join(root, "osediff", clip, "%08d.png" % fid)
            if os.path.exists(dst):
                continue
            src = Image.open(os.path.join(DATA, "LQ", clip,
                                          "%08d.png" % fid)).convert("RGB")
            w, h = src.size
            up = src.resize((w * 4, h * 4), Image.LANCZOS)
            up = up.resize((up.width - up.width % 8, up.height - up.height % 8),
                           Image.LANCZOS)
            lq = transforms.ToTensor()(up).unsqueeze(0).cuda() * 2 - 1
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model(lq.half(), prompt="")
            pil = transforms.ToPILImage()(out[0].float().cpu() * 0.5 + 0.5)
            pil = wavelet_color_fix(target=pil, source=up)
            if pil.size != (w * 4, h * 4):
                pil = pil.resize((w * 4, h * 4), Image.LANCZOS)
            wr(dst, np.asarray(pil))
            print("[osediff] %s %08d %d/%d %.1fs"
                  % (clip, fid, k + 1, len(ids), time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=["000", "016", "005"])
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--root", default="/tmp/p3_demo/out")
    ap.add_argument("--which", default="rbv,edvr")
    ap.add_argument("--rbv-run", default="/tmp/rbv_step0.pth")
    ap.add_argument("--b1-run", default=os.path.join(M2B, "runs/B1_uniform/last.pth"))
    args = ap.parse_args()

    want = [w.strip() for w in args.which.split(",") if w.strip()]
    if "rbv" in want:
        run_rbv(args.root, args.clips, args.frames, args.rbv_run)
    if "b1" in want:
        run_rbv(args.root, args.clips, args.frames, args.b1_run)
    if "edvr" in want:
        run_edvr(args.root, args.clips, args.frames)
    if "osediff" in want:
        run_osediff(args.root, args.clips, args.frames)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
