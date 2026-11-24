# -*- coding: utf-8 -*-
"""Build the comparison sheet from explicit (clip, frame, crop) picks.

The texture-oriented picks below were chosen by eye from a
highest-gradient-crop sweep, so each row shows what the study actually needs to
judge: foliage (016, 019) and brick/gravel texture (011).  Explicit coordinates
beat an automatic picker here because the automatic one kept landing on faces.

Columns: GT | EDVR | RealBasicVSR | OSEDiff | RealBasicVSR->OSEDiff
Each non-reference tile carries its PSNR against GT for that exact crop.

Usage:
  python make_compare_picks.py --out compare_textures.png
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("RBVSR_DATA") or os.path.join(ROOT, "data", "reds_real_c9")
OUT = os.path.join(ROOT, "outputs")

COLS = [
    ("GT", None),
    ("EDVR", "realvsr/edvr_c9/{clip}/{fid}.png"),
    ("RealBasicVSR", "reds_real_c9/gen_thres255_generator/{clip}/{fid}.png"),
    ("OSEDiff", "realvsr/osediff_c9/{clip}/{fid}.png"),
    ("RBV+OSEDiff", "reds_real_c9/rbv_osediff/{clip}/{fid}.png"),
]

# (clip, frame, x, y, what-it-shows)
PICKS = [
    ("016", 22, 932, 116, "foliage: fine leaf texture"),
    ("000", 24, 196, 20, "brick facade + railing, repeating pattern"),
    ("005", 24, 756, 100, "scaffold grid, thin structures"),
]

CROP = 160
ZOOM = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


def read_rgb(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def psnr_luma(a, b):
    ya = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    yb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    mse = float(np.mean((ya - yb) ** 2))
    return 99.0 if mse <= 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def fit_scale(text, scale, thick, max_width, x0):
    """Shrink `scale` until `text` fits, then draw once (never twice)."""
    if not max_width:
        return scale
    avail = max_width - x0 - 4
    for _ in range(60):
        if cv2.getTextSize(text, FONT, scale, thick)[0][0] <= avail:
            break
        if scale <= 0.22:
            break
        scale -= 0.01
    return scale


def put(img, text, org, scale=0.5, thick=1, max_width=None):
    """One pass only: thick antialiased text is broken on OpenCV 5.0 here."""
    s = fit_scale(text, scale, thick, max_width, org[0])
    cv2.putText(img, text, org, FONT, s, (255, 255, 255), thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="compare_textures.png")
    ap.add_argument("--crop", type=int, default=CROP)
    ap.add_argument("--zoom", type=int, default=ZOOM)
    args = ap.parse_args()

    rows = []
    for clip, fid, x, y, what in PICKS:
        tag = "%08d" % fid
        gt = read_rgb(os.path.join(DATA, "GT", clip, tag + ".png"))
        tiles = []
        for label, tmpl in COLS:
            if tmpl is None:
                im = gt
            else:
                p = os.path.join(OUT, tmpl.format(clip=clip, fid=tag))
                if not os.path.exists(p):
                    raise SystemExit("missing %s for %s/%s" % (p, clip, tag))
                im = read_rgb(p)
            c = im[y:y + args.crop, x:x + args.crop]
            if args.zoom > 1:
                c = cv2.resize(c, (args.crop * args.zoom,) * 2,
                               interpolation=cv2.INTER_NEAREST)
            c = c.copy()
            head = ("GT" if label == "GT"
                    else "%s %.2f" % (label, psnr_luma(im, gt)))
            cv2.rectangle(c, (0, 0), (c.shape[1] - 1, 24), (0, 0, 0), -1)
            put(c, head, (5, 17), 0.45, max_width=c.shape[1])
            tiles.append(c)
        row = np.hstack(tiles)

        strip = np.full((26, row.shape[1], 3), 40, np.uint8)
        put(strip, "clip %s  frame %s  %dpx crop  x=%d y=%d  |  %s"
            % (clip, tag, args.crop, x, y, what), (6, 18), 0.55,
            max_width=strip.shape[1])
        rows.append((strip, row))
        print("row %s/%s %s" % (clip, tag, what))

    canvas = np.vstack([r for pair in rows for r in pair])
    dst = os.path.join(ROOT, "results", "reds_real_c9", args.out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print("wrote", dst, canvas.shape)


if __name__ == "__main__":
    main()
