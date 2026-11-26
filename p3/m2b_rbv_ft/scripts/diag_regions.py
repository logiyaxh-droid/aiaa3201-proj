# -*- coding: utf-8 -*-
"""Region-resolved diagnosis: flat vs textured, on the full 30-clip set.

Why this exists
---------------
The whole point of Direction C is that *where* you spend the perceptual budget
matters: flat regions should stay clean and temporally locked, textured regions
should get the rendering freedom.  Whole-image means cannot show that, and on the
3-clip dev set B2 (uncertain) was marginally worse than B1 (uniform) everywhere,
which left the mechanism untested rather than refuted.

Region definition
-----------------
Weight maps are rebuilt exactly as in training from the frozen reference model:
`u = U_1` normalised by its own 95th percentile.  Then
  flat     : u <= 0.25      (bottom quartile of uncertainty)
  textured : u >= 0.75      (top quartile)
This is a FIXED rule applied identically to every method, so the partition does
not favour either arm.

Metrics per region
------------------
PSNR and E_warp are computed on the masked pixels only; E_warp reuses the same
occlusion-aware convention as score_reds_real.py (sum-consistency test).
Noise is measured as the local high-frequency energy in flat regions (Laplacian
stdev on a 3x3 kernel), which is the quantity that over-sharpening inflates.

Usage:
  python diag_regions.py --baseline /tmp/rbv_step0.pth \
      --arms b1=runs/B1_uniform/last.pth b2=runs/B2_uncertain/last.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
M2B = os.path.dirname(HERE)
sys.path.insert(0, M2B)

import warp as W                                   # noqa: E402
from model_rbv_ft import UARBVSR                   # noqa: E402
from train_rbv_ft import load_ckpt                 # noqa: E402

DATA = "/data/disk1/home/ghe270/p3/prior_qualification/data/reds_real_c9"
FLAT_T = 0.25
TEX_T = 0.75


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def psnr_mask(a, b, mask):
    ya = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    yb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    d = (ya - yb) ** 2
    v = d[mask]
    if v.size == 0:
        return None
    mse = float(v.mean())
    return 99.0 if mse <= 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def e_warp_mask(prev, cur, mask, tol=3.0):
    ga = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
    f_fwd = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    f_bwd = cv2.calcOpticalFlowFarneback(gb, ga, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = ga.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    bx = np.clip(gx + f_fwd[..., 0], 0, w - 1).astype(np.float32)
    by = np.clip(gy + f_fwd[..., 1], 0, h - 1).astype(np.float32)
    back = cv2.remap(f_bwd, bx, by, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    fb = np.abs(f_fwd + back).sum(axis=2)
    warped = cv2.remap(cur, gx + f_fwd[..., 0], gy + f_fwd[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    valid = (fb <= tol) & mask
    if not valid.any():
        return None
    diff = np.abs(warped.astype(np.float32) - prev.astype(np.float32)).mean(axis=2)
    return float(diff[valid].mean())


def hf_noise(img, mask):
    """Local high-frequency energy inside the mask (flat-region cleanliness)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    v = np.abs(lap)[mask]
    return float(v.mean()) if v.size else None


def region_masks(model, lq_tensor):
    """U_1 from the frozen reference on the full frame, then threshold."""
    with torch.no_grad():
        b, bf = model.reference_forward(lq_tensor)
        u = W.uncertainty_u1(b, bf)          # [N,1,H,W] on the HR grid
    u = u.mean(dim=0)[0].float().cpu().numpy()
    u = cv2.resize(u, (lq_tensor.shape[-1] * 4, lq_tensor.shape[-2] * 4),
                   interpolation=cv2.INTER_NEAREST)
    return (u <= FLAT_T), (u >= TEX_T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="name=path pairs")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--chunk", type=int, default=9)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = UARBVSR("cuda")
    clips = args.clips or sorted(d for d in os.listdir(os.path.join(DATA, "GT"))
                                 if os.path.isdir(os.path.join(DATA, "GT", d)))

    runs = [("baseline", args.baseline)]
    for spec in args.arms:
        n, p = spec.split("=", 1)
        runs.append((n, p))

    acc = {n: {"flat": [], "tex": []} for n, _ in runs}
    masks_by_clip = {}

    for name, path in runs:
        load_ckpt(path, model)
        model.net.eval()
        print("[run] %s (%s)" % (name, path), flush=True)
        for clip in clips:
            lq_dir = os.path.join(DATA, "LQ", clip)
            gt_dir = os.path.join(DATA, "GT", clip)
            ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(lq_dir)
                         if f.lower().endswith(".png"))
            lq = np.stack([rd(os.path.join(lq_dir, "%08d.png" % i)) for i in ids])
            gt = np.stack([rd(os.path.join(gt_dir, "%08d.png" % i)) for i in ids])
            t = torch.from_numpy(lq).permute(0, 3, 1, 2).float().cuda() / 255.
            with torch.no_grad():
                y = model.infer_sequence(t, chunk=args.chunk)
            pred = [(y[k].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
                     * 255.0 + 0.5).astype(np.uint8) for k in range(y.shape[0])]
            if clip not in masks_by_clip:
                masks_by_clip[clip] = region_masks(model, t)
            fm, tm = masks_by_clip[clip]

            for k in range(len(pred)):
                for tag, m in (("flat", fm), ("tex", tm)):
                    ps = psnr_mask(pred[k], gt[k], m)
                    nz = hf_noise(pred[k], m)
                    ns = hf_noise(gt[k], m)
                    rec = {"psnr": ps, "hf": nz, "hf_gt": ns}
                    if k > 0:
                        rec["e_warp"] = e_warp_mask(pred[k - 1], pred[k], m)
                    acc[name][tag].append(rec)
        print("   done %d clips" % len(clips), flush=True)

    out = {}
    for name, _ in runs:
        out[name] = {}
        for tag in ("flat", "tex"):
            rows = acc[name][tag]
            def m(key):
                v = [r[key] for r in rows if r.get(key) is not None]
                return float(np.mean(v)) if v else None
            out[name][tag] = {"psnr": m("psnr"), "e_warp": m("e_warp"),
                              "hf": m("hf"), "hf_gt": m("hf_gt"),
                              "n": len(rows)}

    print("\n%-12s %-6s %8s %9s %9s %9s" %
          ("method", "region", "PSNR", "E_warp", "HF", "HF_gt"))
    for name, _ in runs:
        for tag in ("flat", "tex"):
            r = out[name][tag]
            def f(v, w=9, p=4):
                return "n/a".rjust(w) if v is None else ("%*.4f" % (w, v))
            print("%-12s %-6s %8s %9s %9s %9s"
                  % (name, tag, f(r["psnr"], 8), f(r["e_warp"]),
                     f(r["hf"]), f(r["hf_gt"])))

    dst = os.path.join(M2B, "results", "regions.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump({"flat_threshold": FLAT_T, "tex_threshold": TEX_T,
                   "n_clips": len(clips), "results": out}, fh,
                  indent=1, ensure_ascii=False)
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
