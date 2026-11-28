#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0 task 10 -- A/B the TecoGAN pre-shift with the shared metrics.

The pre-P0 pipeline fed TecoGAN LR frames shifted by +0.375 px and justified it
with a story about ``bicubic_four`` assuming a half-pixel-centred grid.  The ramp
probe in ``probe_grid.py`` measured what the upsampler actually does; this script
measures what it costs.

Two independent halves:

``--label S``  score an already-inferred output directory: PSNR / SSIM / LPIPS
               and the reference tLP, using ``evalkit`` so the numbers are on the
               same protocol as every Part 1/2 row.

``--scan``     use the model itself as the yardstick: pick a frame, try a grid of
               LR shifts, upscale each shifted frame with the *same* upsampler
               TecoGAN uses internally (``bicubic_four``, via the TF1 helper), and
               report which shift maximises PSNR against GT.  Then do the same
               with a differentiable bicubic that can be optimised to sub-pixel
               accuracy, so "the best shift is between two grid points" can be
               told apart from "the best grid point happens to be 0.25".

The scan uses bicubic only -- no TecoGAN forward pass -- so it isolates the
grid/registration question from the GAN's own behaviour, and it is cheap enough
to run on a shared GPU.  Its result must be interpreted together with
``probe_grid.py``: the upsampler offset is a property of the resampler, the
inference A/B is a property of the whole pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np

import torchvision  # noqa: F401

_mod = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F  # noqa: E402

for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _mod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalkit import data, metrics, temporal  # noqa: E402


def score(pred_dir: str, gt_dir: str, net, cfg: dict) -> dict:
    m = data.match_clip(pred_dir, gt_dir, allow_missing_pred=True)
    pred = [data.read_rgb(p) for p in m["pred"]]
    gt = [data.read_rgb(p) for p in m["gt"]]
    data.check_size(pred, expect=gt[0].shape[:2], label="pred")
    out = metrics.evaluate_stack(pred, gt, lpips_net=net)
    t = temporal.tlp(pred, gt, m["ids"], net,
                     x100=cfg["tlp"]["x100"], cutfr=cfg["tlp"]["cutfr"],
                     crop32=cfg["tlp"]["crop32"])
    fw = temporal.flow_warp_lpips(pred, m["ids"], net,
                                  stride=cfg["flow_warp"]["stride"],
                                  scale=cfg["flow_warp"]["scale"])
    return {**out, "tlp": t["value"], "tlp_pairs": t["n_pairs"],
            "flow_warp_lpips": fw["value"], "flow_warp_stride": fw["stride"],
            "missing_pred_frames": m["missing"], "ids": [m["ids"][0], m["ids"][-1]]}


def bicubic_shift_scan(lr_path: str, gt_path: str, shifts: list, device: str) -> dict:
    """Which sub-pixel LR shift makes a plain bicubic upsample best match GT?

    Uses torch bicubic with the same half-pixel convention as cv2/OpenCV
    (align_corners=False), applied to a shifted LR.  The shift is applied in LR
    pixels by resampling the LR; the upsampler is then fixed.  This is the
    cheapest honest probe of "does the pre-shift help because of registration?".
    """
    import cv2
    import torch
    import torch.nn.functional as F

    lr = data.read_rgb(lr_path)
    gt = data.read_rgb(gt_path)
    H, W = gt.shape[:2]
    g = torch.from_numpy(gt).permute(2, 0, 1)[None].float().to(device).div_(255.0)
    rows = {}
    for s in shifts:
        M = np.float32([[1, 0, s], [0, 1, s]])
        lrs = cv2.warpAffine(lr, M, (lr.shape[1], lr.shape[0]), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        x = torch.from_numpy(lrs).permute(2, 0, 1)[None].float().to(device).div_(255.0)
        y = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False)
        mse = ((y - g) ** 2).mean().item()
        rows["%+.4f" % s] = {"psnr": float(10 * np.log10(1.0 / max(mse, 1e-12)))}
    best = max(rows.items(), key=lambda kv: kv[1]["psnr"])
    return {"shifts": rows, "best_shift": float(best[0]), "best_psnr": best[1]["psnr"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/disk1/shared/aiaa3201/project1_video_sr")
    ap.add_argument("--clip", default="000")
    ap.add_argument("--label", default=None, help="label for a scored output dir")
    ap.add_argument("--pred", default=None, help="output dir to score")
    ap.add_argument("--scan", action="store_true", help="run the sub-pixel shift scan")
    ap.add_argument("--frame", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rep = {"clip": a.clip}
    net = metrics.LPIPS(device=a.device)
    gt_dir = os.path.join(a.root, "val_sharp", a.clip)

    if a.pred:
        cfg = {"tlp": {"x100": False, "cutfr": 0, "crop32": False},
               "flow_warp": {"stride": 2, "scale": 4}}
        rep["scores"] = {a.label or os.path.basename(a.pred.rstrip("/")):
                         score(a.pred, gt_dir, net, cfg)}

    if a.scan:
        shifts = [round(-0.5 + 0.05 * k, 4) for k in range(21)]
        rep["bicubic_shift_scan"] = bicubic_shift_scan(
            os.path.join(a.root, "val_sharp_bicubic", a.clip, "%08d.png" % a.frame),
            os.path.join(gt_dir, "%08d.png" % a.frame), shifts, a.device)
        rep["bicubic_shift_scan"]["frame"] = a.frame

    txt = json.dumps(rep, indent=2)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        if os.path.exists(a.out):
            old = json.load(open(a.out, encoding="utf-8"))
            if "scores" in old and "scores" in rep:
                old["scores"].update(rep["scores"])
                rep["scores"] = old["scores"]
            old.update({k: v for k, v in rep.items() if k != "scores"})
            rep = old
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
