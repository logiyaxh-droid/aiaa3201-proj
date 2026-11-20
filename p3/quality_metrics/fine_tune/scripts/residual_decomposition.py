# -*- coding: utf-8 -*-
"""Did the perceptual objective learn texture synthesis, or just a sharpening filter?

A fine-tuned model's residual over the baseline can be decomposed:

    r = pred_ft - pred_base

If r is well explained by a scalar multiple of the baseline's own high-pass
response, the network learned (at most) an unsharp mask, not new structure.  That
matters because a global unsharp mask raises DISTS/LPIPS while adding ringing in
flat areas -- exactly the pattern the high-frequency diagnostic hinted at.

Reported per clip:
  * cosine similarity between r and each candidate basis
      - baseline high-pass  : "just sharpening"
      - GT high-pass        : "recovering the missing detail"
      - GT minus baseline   : "moving toward GT" (normalised direction)
  * R^2 of the least-squares fit  r ~ a * highpass(baseline)
  * the fitted gain a

A model that genuinely synthesises detail will show a materially larger share of
its residual along (GT - baseline) than along highpass(baseline).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = "/data/disk1/home/ghe270/p3/quality_metrics"
sys.path.insert(0, os.path.join(ROOT, "fine_tune", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cv2
import numpy as np
import common as C
import evaluate as E


def highpass(x):
    g = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.Laplacian(g, cv2.CV_32F, ksize=3)


def cosine(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a.dot(b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def analyse(pred_ft, pred_base, gt, box):
    x, y, w, h = box
    a = pred_ft[y:y + h, x:x + w].astype(np.float32)
    b = pred_base[y:y + h, x:x + w].astype(np.float32)
    g = gt[y:y + h, x:x + w].astype(np.float32)

    r = a - b
    hp_b = highpass_vis_source(b)
    hp_g = highpass_vis_source(g)
    direction = g - b

    # least squares r = k * hp_b, on the grey-magnitude basis used by the metric
    k = float((r * hp_b).sum() / max((hp_b * hp_b).sum(), 1e-9))
    resid = r - k * hp_b
    r2 = 1.0 - float((resid ** 2).sum() / max((r ** 2).sum(), 1e-9))

    return {"cos_r_hp_baseline": cosine(r, hp_b),
            "cos_r_hp_gt": cosine(r, hp_g),
            "cos_r_direction_gt_minus_base": cosine(r, direction),
            "sharpening_fit_k": k,
            "sharpening_fit_r2": r2,
            "residual_energy_ratio": float(np.linalg.norm(r) / max(
                np.linalg.norm(direction), 1e-9))}


def highpass_vis_source(rgb_float):
    """High-pass on a float RGB patch, replicated to 3 channels as a basis."""
    g = cv2.cvtColor(np.clip(rgb_float, 0, 255).astype(np.uint8),
                     cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return np.repeat(hp[..., None], 3, axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="name=checkpoint")
    ap.add_argument("--clips", nargs="*", default=["000", "002", "009"])
    ap.add_argument("--frames", nargs="*", type=int, default=[10, 40, 70, 95])
    ap.add_argument("--box", nargs=4, type=int, default=[400, 240, 480, 320])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    name, _, path = args.candidate.partition("=")
    net_ft, meta = E.load_checkpoint(path, args.device)
    net_ft.eval()
    net_base, _ = E.load_checkpoint("baseline", args.device)
    net_base.eval()

    out = {"candidate": name, "checkpoint": meta, "box": args.box, "clips": {}}
    for clip in args.clips:
        ids = C.clip_frame_ids(os.path.join(C.GT_ROOT, clip))
        gt = C.read_clip(os.path.join(C.GT_ROOT, clip), ids)
        lr = C.read_clip(os.path.join(C.LR_ROOT, clip), ids)
        sel = [lr[i] for i in args.frames]
        pf = E.predict(net_ft, sel, args.device)
        pb = E.predict(net_base, sel, args.device)
        rows = [analyse(pf[k], pb[k], gt[i], args.box)
                for k, i in enumerate(args.frames)]
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out["clips"][clip] = {"frames": args.frames, "mean": agg, "per_frame": rows}
        print("%-6s cos(hp_base) %+.3f  cos(hp_gt) %+.3f  cos(GT-base) %+.3f  "
              "fit R2 %.3f  k %.3f  energy %.3f"
              % (clip, agg["cos_r_hp_baseline"], agg["cos_r_hp_gt"],
                 agg["cos_r_direction_gt_minus_base"], agg["sharpening_fit_r2"],
                 agg["sharpening_fit_k"], agg["residual_energy_ratio"]), flush=True)

    C.write_json(args.out, out)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
