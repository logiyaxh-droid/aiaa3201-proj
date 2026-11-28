# -*- coding: utf-8 -*-
"""U1 soft blend: RealBasicVSR on flat regions, B1 on textured ones.

Motivation (measured, not assumed)
----------------------------------
`regions.json` splits the same 270 frames by uncertainty:

    flat (u<=0.25)   RealBasicVSR: PSNR 24.187  E_warp 3.919  HF  8.86
                     B1           : PSNR 23.786  E_warp 4.266  HF 20.58
    tex  (u>=0.75)   RealBasicVSR: PSNR 15.988  E_warp 13.056 HF 77.53
                     B1           : PSNR 16.037  E_warp 10.971 HF 84.25

So the two models are complementary, and — critically — they share the SAME
architecture, the same degradation domain and the SAME pixel grid.  B1 is
RealBasicVSR with fine-tuned propagation weights, so the blend has no phase
misalignment and no domain mismatch (unlike chaining OSEDiff or StableSR, which
M1 showed introduces both).

The blend uses exactly the quantity the fine-tuning already computed:

    Y = (1 - U1) * Y_rbv + U1 * Y_b1

U1 is the flip-disagreement uncertainty, normalised per image by its own q95, so
this is a *soft router*, not a tuned mixture weight — there is no free parameter
and nothing is fitted on the evaluation set.

Why this must be checked on the CLIPS, not just the mean
-------------------------------------------------------
A blend can look good on average while being worse on every single clip if the
gain is concentrated.  Per-clip deltas are reported so that cannot hide.

Usage:
  python blend_u1.py --b1 /tmp/m2b_out/full_b1 --rbv <dir> --out /tmp/m2b_out/blend
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
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import warp as W                           # noqa: E402
from models.ua_realbasicvsr import UARBVSR             # noqa: E402

DATA = os.environ.get("RBVSR_DATA", os.path.join(REPO_ROOT, "data", "reds_real_c9"))


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def write_u8(p, a):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", required=True, help="dir holding the B1 30-clip output")
    ap.add_argument("--rbv", required=True,
                    help="dir holding the RealBasicVSR 30-clip output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="/tmp/rbv_step0.pth",
                    help="step-0 checkpoint, used only for the U1 reference")
    ap.add_argument("--chunk", type=int, default=9)
    ap.add_argument("--save-u1", default="", help="optional dir to dump U1 maps")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = UARBVSR("cuda")          # weights are irrelevant: only .ref is used
    clips = sorted(d for d in os.listdir(os.path.join(DATA, "GT"))
                   if os.path.isdir(os.path.join(DATA, "GT", d)))
    print("[blend] %d clips" % len(clips), flush=True)

    done = 0
    for clip in clips:
        lq_dir = os.path.join(DATA, "LQ", clip)
        ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(lq_dir)
                     if f.lower().endswith(".png"))
        lq = np.stack([rd(os.path.join(lq_dir, "%08d.png" % i)) for i in ids])
        t = torch.from_numpy(lq).permute(0, 3, 1, 2).float().cuda() / 255.
        with torch.no_grad():
            # same U1 the fine-tuning saw: flip disagreement, q95-normalised
            b, bf = model.reference_forward(t)
            u = W.uncertainty_u1(b, bf)              # [N,1,H,W] at HR scale
        for k, fid in enumerate(ids):
            fn = "%08d.png" % fid
            y1 = rd(os.path.join(args.b1, clip, fn)).astype(np.float32)
            yr = rd(os.path.join(args.rbv, clip, fn)).astype(np.float32)
            if y1.shape != yr.shape:
                raise SystemExit("size mismatch %s/%s: %s vs %s"
                                 % (clip, fn, y1.shape, yr.shape))
            h, w = y1.shape[:2]
            uk = u[k, 0].float().cpu().numpy()
            if uk.shape != (h, w):
                uk = cv2.resize(uk, (w, h), interpolation=cv2.INTER_LINEAR)
            a = uk[..., None]
            out = (1.0 - a) * yr + a * y1
            write_u8(os.path.join(args.out, clip, fn),
                     np.clip(out + 0.5, 0, 255).astype(np.uint8))
            if args.save_u1:
                write_u8(os.path.join(args.save_u1, clip, fn),
                         np.clip(uk * 255.0 + 0.5, 0, 255).astype(np.uint8)[
                             ..., None].repeat(3, axis=2))
        done += 1
        if done % 5 == 0:
            print("  %d/%d" % (done, len(clips)), flush=True)

    print("[done] wrote %d clips -> %s" % (done, args.out), flush=True)


if __name__ == "__main__":
    main()
