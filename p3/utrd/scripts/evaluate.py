# -*- coding: utf-8 -*-
"""Evaluate one prototype arm and the frozen EDVR baseline on fixed dev frames.

Protocol is inherited from the proven pipeline: RGB uint8 PNG input, full frame,
no crop, no resize, Y-PSNR as in p0/evalkit.  Only the frame subset changes:
development clips 000/002/009, frames 40..69 (30 frames each), plus a fixed
tLP over those consecutive pairs.

Usage:
  venv/bin/python scripts/evaluate.py --mode det --name det --frames 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")

import numpy as np
import torch
import torch.nn.functional as F

import model as M
import train as T
from evalkit import metrics as EM
from evalkit import temporal as ET

VAL = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp"
VAL_LR = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp_bicubic"
DEV_CLIPS = ("000", "002", "009")
FRAMES = (40, 70)


def read_seq(clip, start, end):
    ids = T.frame_ids(os.path.join(VAL, clip))
    sel = [i for i in ids if start <= i < end]
    gt = [T.read_rgb(os.path.join(VAL, clip, "%08d.png" % i)) for i in sel]
    lr = [T.read_rgb(os.path.join(VAL_LR, clip, "%08d.png" % i)) for i in sel]
    return sel, lr, gt


def infer_arm(net, mode, lr_list, edvr):
    """Centre-frame output for each frame in the list, five-frame windows."""
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().div(255).permute(2, 0, 1)
    n = len(lr_list)
    preds = []
    for c in range(n):
        win = [max(0, min(n - 1, c + d - 2)) for d in range(5)]
        lr = torch.stack([to_t(lr_list[w]) for w in win])[None].cuda()
        base, feat, u = M.edvr_outputs(edvr, lr)
        lr_up = F.interpolate(lr[:, 2], scale_factor=4, mode="bilinear",
                              align_corners=False)
        out = M.forward_one(net, mode, base, lr_up, feat, u, seed=1234)
        preds.append(out[0].mul(255).round().clamp(0, 255).byte()
                     .permute(1, 2, 0).cpu().numpy())
    return preds


def baseline_only(lr_list, edvr):
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().div(255).permute(2, 0, 1)
    preds = []
    n = len(lr_list)
    for c in range(n):
        win = [max(0, min(n - 1, c + d - 2)) for d in range(5)]
        lr = torch.stack([to_t(lr_list[w]) for w in win])[None].cuda()
        base, _, _ = M.edvr_outputs(edvr, lr)
        preds.append(base[0].mul(255).round().clamp(0, 255).byte()
                     .permute(1, 2, 0).cpu().numpy())
    return preds


def score(pred, gt, ids):
    pY = np.stack([EM.luma(np.asarray(x)) for x in pred])
    gY = np.stack([EM.luma(np.asarray(x)) for x in gt])
    psnr = float(EM.psnr_mean(pY, gY))
    ssim = float(EM.ssim_gpu(pY, gY, device="cuda").mean())
    lp = EM.LPIPS(device="cuda", torch_home="/tmp/torch_home")
    lpips = float(lp.pair(np.stack(pred), np.stack(gt)).mean())
    pairs = ET.pairs_of(ids, 0)
    if pairs:
        d_sr = lp.pair(np.stack([pred[a] for a, _ in pairs]),
                       np.stack([pred[b] for _, b in pairs]))
        d_gt = lp.pair(np.stack([gt[a] for a, _ in pairs]),
                       np.stack([gt[b] for _, b in pairs]))
        tlp = float(np.abs(d_sr - d_gt).mean())
    else:
        tlp = None
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips, "tlp": tlp,
            "n_frames": len(ids), "n_pairs": len(pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("baseline",) + M.MODES, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--start", type=int, default=FRAMES[0])
    ap.add_argument("--end", type=int, default=FRAMES[1])
    ap.add_argument("--clips", nargs="*", default=list(DEV_CLIPS))
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    edvr = M.FrozenEDVR("cuda")
    net = None
    if args.mode != "baseline":
        net = M.ResidualUNet(base=64).cuda()
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        net.eval()

    out_dir = os.path.join(ROOT, "results", args.name)
    os.makedirs(out_dir, exist_ok=True)
    results = {"name": args.name, "mode": args.mode,
               "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
               "clips": {}, "frames": [args.start, args.end],
               "protocol": {"psnr": EM.CONVENTIONS["psnr"],
                            "lpips": "LPIPS Alex v0.1, RGB uint8, full frame",
                            "tlp": ET.TLP_DEFINITION}}
    for clip in args.clips:
        ids, lr, gt = read_seq(clip, args.start, args.end)
        pred = (baseline_only(lr, edvr) if args.mode == "baseline"
                else infer_arm(net, args.mode, lr, edvr))
        row = score(pred, gt, ids)
        results["clips"][clip] = row
        print("%s %s PSNR %.4f SSIM %.4f LPIPS %.5f tLP %s (%d frames)"
              % (args.name, clip, row["psnr"], row["ssim"], row["lpips"],
                 ("%.5f" % row["tlp"]) if row["tlp"] is not None else "N/A",
                 row["n_frames"]), flush=True)
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    for k in ("psnr", "ssim", "lpips", "tlp"):
        vals = [v[k] for v in results["clips"].values() if v[k] is not None]
        results.setdefault("mean", {})[k] = float(np.mean(vals)) if vals else None
    results["mean"]["n_clips"] = len(results["clips"])
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("mean:", json.dumps(results["mean"]), flush=True)


if __name__ == "__main__":
    main()
