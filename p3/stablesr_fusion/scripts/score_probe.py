#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score the 40-frame screen: fixed blends, two oracles, spatial and temporal.

Reads the caches produced by cache_edvr_probe.py and cache_stablesr_probe.py and
writes one JSON plus a markdown summary. No training, no inference: this only
measures whether the generation branch carries usable complementary signal.

Metrics follow P0 exactly: Y of cv2.COLOR_RGB2YCrCb, uint8, no crop, per-frame
PSNR then mean; RGB-MSE in 0-255 units; tLP over the 9 consecutive pairs per clip.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")
sys.path.insert(0, "/data/disk2/home/ghe270/p0")

import cv2
import numpy as np
import torch

import types
import torchvision
from torchvision.transforms import functional as _F

_mod = types.ModuleType("torchvision.transforms.functional_tensor")
for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _mod
torch.backends.cudnn.benchmark = False

from evalkit import metrics as M
from evalkit import temporal as T

ROOT = "/data/disk2/home/ghe270/p3/stablesr_fusion/probe"
BASE = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = os.path.join(BASE, "train_sample_sharp")
CLIPS = ["016", "017", "018", "019"]
FRAMES = list(range(40, 50))
DECW = [0.0, 0.5, 1.0]
ALPHAS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
BLOCK = 32


def read_rgb(path):
    a = cv2.imread(path)
    if a is None:
        raise RuntimeError("unreadable %s" % path)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def metrics(pred, gt, ids, lpips_net):
    pY = np.stack([M.luma(x) for x in pred])
    gY = np.stack([M.luma(x) for x in gt])
    out = {
        "psnr": float(M.psnr_mean(pY, gY)),
        "ssim": float(M.ssim_gpu(pY, gY, device="cuda").mean()),
        "lpips": lpips_net.mean(np.stack(pred), np.stack(gt)),
        "rgb_mse": float(np.mean([((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
                                  for a, b in zip(pred, gt)])),
    }
    t = T.tlp(pred, gt, ids, lpips_net, x100=False, cutfr=0, crop32=False)
    out["tlp"] = t["value"]
    out["tlp_pairs"] = t["n_pairs"]
    return out


def block_oracle(E, G, GT, block=BLOCK):
    """Per-block choice of E or G by RGB MSE, plus per-block continuous alpha."""
    h, w = E.shape[:2]
    out_choice = E.copy()
    out_alpha = np.zeros_like(E, dtype=np.float64)
    for y in range(0, h, block):
        for x in range(0, w, block):
            sl = (slice(y, min(y + block, h)), slice(x, min(x + block, w)))
            e = E[sl].astype(np.float64)
            g = G[sl].astype(np.float64)
            gt = GT[sl].astype(np.float64)
            if ((e - gt) ** 2).mean() <= ((g - gt) ** 2).mean():
                out_choice[sl] = E[sl]
            else:
                out_choice[sl] = G[sl]
            d = (g - e).ravel()
            num = float(((gt - e).ravel() * d).sum())
            den = float((d * d).sum()) + 1e-12
            a = min(max(num / den, 0.0), 1.0)
            out_alpha[sl] = e + a * (g - e)
    return (np.clip(np.rint(out_choice), 0, 255).astype(np.uint8),
            np.clip(np.rint(out_alpha), 0, 255).astype(np.uint8))


def main():
    lpips_net = M.LPIPS(device="cuda", torch_home="/tmp/torch_home")
    result = {"protocol": {"clips": CLIPS, "frames": FRAMES, "dec_w": DECW,
                           "alphas": ALPHAS, "block": BLOCK},
              "per_clip": {}, "gate": {}}
    for clip in CLIPS:
        E = [read_rgb(os.path.join(ROOT, "cache", "edvr", clip, "%08d.png" % f))
             for f in FRAMES]
        GT = [read_rgb(os.path.join(GT_ROOT, clip, "%08d.png" % f)) for f in FRAMES]
        assert all(e.shape == g.shape for e, g in zip(E, GT)), clip
        row = {}
        row["E"] = metrics(E, GT, FRAMES, lpips_net)
        for decw in DECW:
            G = [read_rgb(os.path.join(ROOT, "cache", "stablesr",
                                       "decw_%.1f" % decw, clip, "%08d.png" % f))
                 for f in FRAMES]
            if any(g.shape != e.shape for g, e in zip(G, E)):
                raise RuntimeError("shape mismatch clip %s decw %.1f" % (clip, decw))
            tag = "%.1f" % decw
            row["G_" + tag] = metrics(G, GT, FRAMES, lpips_net)
            for a in ALPHAS:
                blended = [np.clip(np.rint(e.astype(np.float64)
                                           + a * (g.astype(np.float64) - e)), 0, 255)
                           .astype(np.uint8) for e, g in zip(E, G)]
                row["M_%s_a%.2f" % (tag, a)] = metrics(blended, GT, FRAMES, lpips_net)
            ch, al = [], []
            for e, g, gt in zip(E, G, GT):
                c, q = block_oracle(e, g, gt)
                ch.append(c)
                al.append(q)
            row["oracle_choice_" + tag] = metrics(ch, GT, FRAMES, lpips_net)
            row["oracle_alpha_" + tag] = metrics(al, GT, FRAMES, lpips_net)
        result["per_clip"][clip] = row
        print("scored", clip, flush=True)

    # aggregate
    def mean_of(key):
        vals = [result["per_clip"][c][key]["psnr"]
                for c in CLIPS if key in result["per_clip"][c]]
        return float(np.mean(vals)) if vals else None

    agg = {}
    keys = sorted({k for c in CLIPS for k in result["per_clip"][c]})
    for k in keys:
        rows = [result["per_clip"][c][k] for c in CLIPS if k in result["per_clip"][c]]
        agg[k] = {m: float(np.mean([r[m] for r in rows]))
                  for m in ("psnr", "ssim", "lpips", "rgb_mse", "tlp")}
    result["mean"] = agg

    # gate: best fixed blend vs E, per dec_w, plus oracle and clip-direction count
    base = agg["E"]
    gate = {"d_psnr_E": 0.0, "d_mse_E": 0.0}
    best = None
    for decw in DECW:
        tag = "%.1f" % decw
        cands = [(a, agg["M_%s_a%.2f" % (tag, a)]) for a in ALPHAS if a > 0]
        for a, m in cands:
            d = m["psnr"] - base["psnr"]
            if best is None or d > best["d_psnr"]:
                best = {"dec_w": decw, "alpha": a, "d_psnr": d,
                        "d_mse": m["rgb_mse"] - base["rgb_mse"],
                        "d_lpips": m["lpips"] - base["lpips"],
                        "d_tlp": m["tlp"] - base["tlp"]}
        oracle = agg["oracle_alpha_" + tag]
        gate["oracle_%s_d_psnr" % tag] = oracle["psnr"] - base["psnr"]
        gate["oracle_choice_%s_d_psnr" % tag] = (
            agg["oracle_choice_" + tag]["psnr"] - base["psnr"])
    gate["best_fixed"] = best
    # per-clip direction of the best fixed setting
    b = best
    tag = "%.1f" % b["dec_w"]
    key = "M_%s_a%.2f" % (tag, b["alpha"])
    gate["clips_positive"] = sum(
        1 for c in CLIPS
        if result["per_clip"][c][key]["psnr"] > result["per_clip"][c]["E"]["psnr"])
    gate["pass"] = bool(b["d_psnr"] >= 0.05 and b["d_mse"] < 0 and gate["clips_positive"] >= 3)
    result["gate"] = gate

    os.makedirs(ROOT, exist_ok=True)
    with open(os.path.join(ROOT, "probe_results.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)

    print(json.dumps({"best_fixed": best, "clips_positive": gate["clips_positive"],
                      "oracle": {k: v for k, v in gate.items() if k.startswith("oracle")},
                      "pass": gate["pass"]}, indent=1))


if __name__ == "__main__":
    main()
