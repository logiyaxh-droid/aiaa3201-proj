# -*- coding: utf-8 -*-
"""Sweep the blend mapping, to separate "U1 is informative" from "U1 as-is".

The raw blend `(1-U1)*RBV + U1*B1` gave the best PSNR and E_warp of any row but
lost most of B1's perceptual gain.  Measured cause: 58.1% of pixels sit at
U1 < 0.25 (median 0.208), while the pixel gap |B1 - RBV| is 16.0/255 in textured
regions vs only 4.1/255 in flat ones.  So the raw weight sends most of the frame
back to RealBasicVSR precisely where B1 had little to say anyway, and *also*
damps B1 where it mattered.

This sweep asks a narrower question: is there a monotone remap of U1 that keeps
the flat-region gain while restoring the textured-region gain?  Every variant is
a fixed, parameter-free rule; nothing is fitted per clip.  The sweep is reported
in full so a lucky setting cannot be presented as a discovery.

  raw      u                                   (the guide's proposal)
  sq       u^2                                 (sharpen: push low u toward 0)
  sqrt     sqrt(u)                             (soften: lift low u)
  norm     (u - p10) / (p90 - p10), clipped    (use the actual dynamic range)
  binary   (u >= 0.5)                          (hard router)
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def rd(p, gray=False):
    a = cv2.imread(p, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return a if gray else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def weight(u, mode, p10=None, p90=None):
    if mode == "raw":
        return u
    if mode == "sq":
        return u * u
    if mode == "sqrt":
        return np.sqrt(u)
    if mode == "norm":
        return np.clip((u - p10) / max(1e-6, p90 - p10), 0, 1)
    if mode == "binary":
        return (u >= 0.5).astype(np.float32)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u1", default="/tmp/m2b_out/u1")
    ap.add_argument("--b1", default="/tmp/m2b_out/full_b1")
    ap.add_argument("--rbv", default="/tmp/m2b_out/full_rbv")
    ap.add_argument("--out-root", default="/tmp/m2b_out")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--modes", nargs="*",
                    default=["raw", "sq", "sqrt", "norm", "binary"])
    args = ap.parse_args()

    clips = args.clips or sorted(d for d in os.listdir(args.u1)
                                 if os.path.isdir(os.path.join(args.u1, d)))
    # dynamic range of U1 across the whole set, for the `norm` mode
    sample = []
    for clip in clips[:5]:
        for f in sorted(os.listdir(os.path.join(args.u1, clip)))[:3]:
            sample.append(rd(os.path.join(args.u1, clip, f), gray=True).ravel())
    s = np.concatenate(sample).astype(np.float32) / 255.
    p10, p90 = float(np.percentile(s, 10)), float(np.percentile(s, 90))
    print("U1 p10 %.4f p90 %.4f" % (p10, p90), flush=True)

    for mode in args.modes:
        dst_root = os.path.join(args.out_root, "blend_%s" % mode)
        for clip in clips:
            for f in sorted(os.listdir(os.path.join(args.u1, clip))):
                u = rd(os.path.join(args.u1, clip, f), gray=True).astype(np.float32) / 255.
                b1 = rd(os.path.join(args.b1, clip, f)).astype(np.float32)
                rb = rd(os.path.join(args.rbv, clip, f)).astype(np.float32)
                w = weight(u, mode, p10, p90)[..., None]
                out = (1.0 - w) * rb + w * b1
                p = os.path.join(dst_root, clip, f)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                cv2.imwrite(p, cv2.cvtColor(
                    np.clip(out + 0.5, 0, 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR))
        print("wrote %s" % dst_root, flush=True)


if __name__ == "__main__":
    main()
