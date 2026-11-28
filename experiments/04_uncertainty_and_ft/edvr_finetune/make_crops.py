# -*- coding: utf-8 -*-
"""Build side-by-side crops so the perceptual question is settled by looking.

For a clip, frame and crop box we write GT | baseline | candidate at 4x zoom,
plus a high-pass residual row (same layout) so added energy is visible directly.

Usage:
    python make_crops.py --clip 000 --frame 40 --box 420 260 260 200 \
        --checkpoint candidate=fine_tune/runs/pilot_dists010/step_000400.pth
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = "/data/disk1/home/ghe270/p3/quality_metrics"
sys.path.insert(0, os.path.join(ROOT, "fine_tune", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cv2
import numpy as np
import common as C
import evaluate as E


def highpass_vis(x, gain=4.0):
    g = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    v = np.clip(128 + hp * gain, 0, 255).astype(np.uint8)
    return np.repeat(v[..., None], 3, axis=2)


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="000")
    ap.add_argument("--frame", type=int, default=40)
    ap.add_argument("--box", nargs=4, type=int, default=[420, 260, 260, 200],
                    help="x y w h in HR pixels")
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="name=path (name=baseline allowed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    x, y, w, h = args.box
    gt_dir = os.path.join(C.GT_ROOT, args.clip)
    ids = C.clip_frame_ids(gt_dir)
    gt = C.read_clip(gt_dir, ids)[args.frame]
    lr = C.read_clip(os.path.join(C.LR_ROOT, args.clip), ids)[args.frame]

    panels_top, panels_bot = [label(gt[y:y + h, x:x + w], "GT")], \
                             [label(highpass_vis(gt[y:y + h, x:x + w]), "GT hp")]
    import torch
    for spec in args.checkpoint:
        name, _, path = spec.partition("=")
        net, _ = E.load_checkpoint(path or "baseline", args.device)
        net.eval()
        pred = E.predict(net, [lr], args.device)[0]
        panels_top.append(label(pred[y:y + h, x:x + w], name))
        panels_bot.append(label(highpass_vis(pred[y:y + h, x:x + w]), name))
        del net
        torch.cuda.empty_cache()

    top = np.concatenate(panels_top, axis=1)
    bot = np.concatenate(panels_bot, axis=1)
    mosaic = np.concatenate([top, bot], axis=0)
    if args.zoom != 1:
        mosaic = cv2.resize(mosaic, None, fx=args.zoom, fy=args.zoom,
                            interpolation=cv2.INTER_NEAREST)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
    print("wrote", args.out, mosaic.shape)


if __name__ == "__main__":
    main()
