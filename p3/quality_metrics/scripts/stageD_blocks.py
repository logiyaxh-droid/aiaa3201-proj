# -*- coding: utf-8 -*-
"""Stage D: turn score differences into located, observable defects.

Stage C answers "which method scores worse".  This stage answers "where, when and
in what kind of region", using only stage C's outputs plus the frames, with no
new model.

D1 regional attribution
    A fixed 32x32 block grid (one block size, chosen once, no multi-scale
    sweep).  For each block: error, high-frequency energy, local gradient
    difference, optical-flow magnitude and forward-backward consistency, and the
    two methods' DISTS difference.
    Block-level DISTS is labelled "局部裁切诊断" everywhere it appears: the
    standard DISTS gives one score per frame, so a block score is a cropped-input
    diagnostic, never the headline number.

D2 motion grouping
    Reuses the existing three-way motion split (p0/out/motion) when present;
    otherwise recomputes per-clip mean flow magnitude and splits at the terciles.
    Reports the clip count and spread per group -- three clips cannot support a
    between-group claim and this is stated in the output.

D3 temporal anomaly location
    For the clips where DOVER technical is worst, lines up the per-pair tLP from
    the stored P0 results, the flow-warp LPIPS, adjacent-frame differences, and
    the worst adjacent pair as a GT / A / B triptych.

D4 metric sensitivity
    MUSIQ second weight (paq2piq) and DOVER repeat sampling seeds, only on the
    representative clips stage D already selected.

Everything writes both a machine-readable JSON and a markdown table so a claim in
the report can be traced to a frame id.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import common as C

CASES = os.path.join(C.ROOT, "cases")
SUMMARY = os.path.join(C.ROOT, "summary")


# ------------------------------------------------------------------ helpers

def load_metric(scope, metric):
    root = os.path.join(C.ROOT, "results", scope, metric)
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*.json"))):
        clip = os.path.splitext(os.path.basename(path))[0]
        rows = json.load(open(path, encoding="utf-8"))
        out[clip] = {m: v for m, v in rows.items() if v.get("status") == "ok"}
    return out


def old_results(clip):
    path = "/data/disk1/home/ghe270/p0/out/full/results/%s.json" % clip
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def flow_between(a_rgb, b_rgb, scale=4):
    """Farneback flow a->b at 1/scale, upsampled back to full resolution."""
    h, w = a_rgb.shape[:2]
    ga = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY)
    sa = cv2.resize(ga, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    sb = cv2.resize(gb, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(sa, sb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return f * float(scale)


def flow_consistency(a_rgb, b_rgb, scale=4):
    """Forward-backward consistency: f_fwd(p) + f_bwd(p + f_fwd(p)) ~ 0.

    Returns (fwd_full, bwd_full, consistency_full), all at full resolution so the
    caller can index them with the same block grid as the image.

    The sign matters.  Written as a subtraction, a perfectly consistent pair
    reports p50 ~14 px and any sane threshold is pushed out of reach, which
    silently disables the mask instead of failing.  ``p0/scripts/align_diag.py``
    documents the same trap; the self-test in ``flow_selftest`` asserts p50 lands
    near 0 for a synthetic pure translation.
    """
    h, w = a_rgb.shape[:2]
    fwd_s = flow_between(a_rgb, b_rgb, scale)            # (h/scale, w/scale, 2)
    bwd_s = flow_between(b_rgb, a_rgb, scale)
    hs, ws = fwd_s.shape[:2]
    gx, gy = np.meshgrid(np.arange(ws, dtype=np.float32),
                         np.arange(hs, dtype=np.float32))
    px = gx + fwd_s[..., 0]
    py = gy + fwd_s[..., 1]
    bx = cv2.remap(bwd_s[..., 0], px, py, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    by = cv2.remap(bwd_s[..., 1], px, py, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    cons_s = np.sqrt((fwd_s[..., 0] + bx) ** 2 + (fwd_s[..., 1] + by) ** 2)

    def up(x):
        return cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)

    fwd = np.dstack([up(fwd_s[..., 0]) * scale, up(fwd_s[..., 1]) * scale])
    bwd = np.dstack([up(bwd_s[..., 0]) * scale, up(bwd_s[..., 1]) * scale])
    return fwd, bwd, up(cons_s) * scale


def flow_selftest():
    """Prove the consistency sign is right: a pure translation must give ~0."""
    a = np.zeros((240, 320, 3), np.uint8)
    a[80:120, 100:140] = 200
    b = cv2.warpAffine(a, np.float32([[1, 0, 5], [0, 1, 3]]), (320, 240),
                       borderMode=cv2.BORDER_REPLICATE)
    fwd, _, cons = flow_consistency(a, b)
    mag = float(np.sqrt(fwd[..., 0] ** 2 + fwd[..., 1] ** 2).mean())
    return {"consistency_p50_px": float(np.median(cons)),
            "flow_mag_mean_px": mag,
            "expected_translation_px": float(np.hypot(5, 3)),
            "pass": bool(np.median(cons) < 1.0)}


def block_grid(h, w, bs=32):
    return [(y, x) for y in range(0, h - bs + 1, bs) for x in range(0, w - bs + 1, bs)]


def block_stats(pred, gt, blocks, bs):
    """Per-block error, high-frequency energy, gradient difference."""
    out = []
    for (y, x) in blocks:
        p = pred[y:y + bs, x:x + bs].astype(np.float32)
        g = gt[y:y + bs, x:x + bs].astype(np.float32)
        diff = p - g
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff ** 2))

        def hf(im):
            gray = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            return float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))))

        def grad(im):
            gray = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            return gx, gy

        pgx, pgy = grad(p)
        ggx, ggy = grad(g)
        gdiff = float(np.mean(np.sqrt((pgx - ggx) ** 2 + (pgy - ggy) ** 2)))
        out.append({"y": y, "x": x, "mae": mae, "mse": mse,
                    "hf_pred": hf(p), "hf_gt": hf(g), "grad_diff": gdiff})
    return out
