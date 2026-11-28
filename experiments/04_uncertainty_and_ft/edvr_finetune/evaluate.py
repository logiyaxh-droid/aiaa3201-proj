# -*- coding: utf-8 -*-
"""Evaluate a fine-tuned EDVR checkpoint using the exact P0/P3 conventions.

The training crop and loss may be new; the evaluation path must not be.  This
script therefore reuses the metric implementations from quality_metrics and the
temporal implementation from p0/evalkit, with the same sliding-window boundary
policy and uint8 rounding as full_eval.py.

A checkpoint may be a full training checkpoint (keys model/step/config) or a
raw BasicSR params dictionary.  Raw model weights are always stored separately
in weights.pth for reproducibility; optimizer/RNG state is never evaluated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = "/data/disk1/home/ghe270/p3/quality_metrics"
sys.path.insert(0, os.path.join(ROOT, "fine_tune", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")

import train as T
import common as C
import numpy as np
import torch
from evalkit import metrics as M
from evalkit import temporal as TT

torch.backends.cudnn.benchmark = False


def load_checkpoint(path, device):
    net = T.build_generator(device)
    if path == "baseline":
        return net, {"name": "baseline", "weights_sha256": C.weight_sha256("edvr_reds")}
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in ck:
        net.load_state_dict(ck["model"], strict=True)
        meta = {"name": os.path.basename(os.path.dirname(path)),
                "checkpoint": os.path.abspath(path), "step": ck.get("step"),
                "config": ck.get("config"),
                "weights_sha256": T.sha256(path)}
    elif "params" in ck:
        net.load_state_dict(ck["params"], strict=True)
        meta = {"name": os.path.basename(path), "checkpoint": path,
                "weights_sha256": T.sha256(path)}
    else:
        raise ValueError("unrecognised checkpoint keys: %s" % list(ck)[:10])
    return net, meta


def predict(net, lr, device):
    out = []
    n = len(lr)
    for c in range(n):
        win = [max(0, min(n - 1, c + d - 2)) for d in range(5)]
        x = torch.stack([torch.from_numpy(lr[w]).permute(2, 0, 1).float().div_(255)
                         for w in win]).unsqueeze(0).to(device)
        with torch.no_grad():
            y = net(x)
        out.append(y[0].clamp_(0, 1).mul_(255).round_().byte()
                   .permute(1, 2, 0).cpu().numpy())
    return out


def score_clip(net, clip, device, dists, lpips_net, musiq=None):
    gt_dir = os.path.join(C.GT_ROOT, clip)
    lr_dir = os.path.join(C.LR_ROOT, clip)
    ids = C.clip_frame_ids(gt_dir)
    if C.clip_frame_ids(lr_dir) != ids:
        raise RuntimeError("LR/GT frame ids differ for %s" % clip)
    gt = C.read_clip(gt_dir, ids)
    lr = C.read_clip(lr_dir, ids)
    pred = predict(net, lr, device)

    pY = np.stack([M.luma(x) for x in pred])
    gY = np.stack([M.luma(x) for x in gt])
    psnr = M.psnr_mean(pY, gY)
    lpips = lpips_net.pair(np.stack(pred), np.stack(gt))
    dv = dists.per_frame(pred, gt)
    pairs = TT.pairs_of(ids, 0)
    d_sr = lpips_net.pair(np.stack([pred[a] for a, _ in pairs]),
                          np.stack([pred[b] for _, b in pairs]))
    d_gt = lpips_net.pair(np.stack([gt[a] for a, _ in pairs]),
                          np.stack([gt[b] for _, b in pairs]))
    out = {"clip": clip, "frame_ids": ids,
           "psnr": float(psnr), "lpips": float(lpips.mean()),
           "dists": float(dv.mean()), "tlp": float(np.abs(d_sr - d_gt).mean()),
           "tlp_pairs": len(pairs),
           "per_frame_lpips": lpips.tolist(), "per_frame_dists": dv.tolist(),
           "per_pair_tlp": np.abs(d_sr - d_gt).tolist()}
    if musiq is not None:
        out["musiq"] = float(musiq.per_frame(pred).mean())
    return out, pred, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="'baseline' or a training checkpoint")
    ap.add_argument("--name", required=True)
    ap.add_argument("--clips", nargs="*", default=["000", "002", "009"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--musiq", action="store_true")
    ap.add_argument("--save-cases", action="store_true")
    args = ap.parse_args()

    net, meta = load_checkpoint(args.checkpoint, args.device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    lpips_net = M.LPIPS(device=args.device)
    dists = C.DistsMetric(device=args.device)
    musiq = C.MusiqMetric(device=args.device) if args.musiq else None

    out_dir = os.path.join(ROOT, "fine_tune", "results", args.name)
    os.makedirs(out_dir, exist_ok=True)
    results = {"name": args.name, "checkpoint": meta,
               "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "protocol": {"psnr": M.CONVENTIONS["psnr"],
                            "lpips": M.CONVENTIONS["lpips"],
                            "dists": dists.pairing(),
                            "tlp": TT.TLP_DEFINITION},
               "clips": {}}
    for clip in args.clips:
        t0 = time.time()
        row, pred, gt = score_clip(net, clip, args.device, dists, lpips_net, musiq)
        results["clips"][clip] = row
        if args.save_cases:
            case_dir = os.path.join(out_dir, "cases", clip)
            os.makedirs(case_dir, exist_ok=True)
            import cv2
            for fid in (0, 50, 99):
                if fid < len(pred):
                    cv2.imwrite(os.path.join(case_dir, "%08d_pred.png" % fid),
                                cv2.cvtColor(pred[fid], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(case_dir, "%08d_gt.png" % fid),
                                cv2.cvtColor(gt[fid], cv2.COLOR_RGB2BGR))
        print("%s clip %s PSNR %.4f LPIPS %.5f DISTS %.5f tLP %.5f (%.1fs)"
              % (args.name, clip, row["psnr"], row["lpips"], row["dists"],
                 row["tlp"], time.time() - t0), flush=True)
        C.write_json(os.path.join(out_dir, "metrics.json"), results)

    vals = {k: [v[k] for v in results["clips"].values()] for k in
            ("psnr", "lpips", "dists", "tlp")}
    results["mean"] = {k: float(np.mean(v)) for k, v in vals.items()}
    results["clip_std"] = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                           for k, v in vals.items()}
    C.write_json(os.path.join(out_dir, "metrics.json"), results)
    print("mean:", json.dumps(results["mean"]), flush=True)


if __name__ == "__main__":
    main()
