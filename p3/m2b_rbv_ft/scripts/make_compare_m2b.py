# -*- coding: utf-8 -*-
"""M2-B comparison sheet: 6 columns at M1's exact crop coordinates.

Columns: GT | EDVR | RealBasicVSR | OSEDiff | Ours_uniform | Ours_uncertain

Crop convention is inherited from M1's make_compare_picks.py: (x, y) is the
TOP-LEFT of a 160px crop, zoomed 2x. Mixing it with a centre-point convention
was what made an earlier sheet show different content in the GT and RBV columns.

Note on cv2.putText: OpenCV 5.0.0 renders LINE_AA with thickness>=3 misaligned,
so labels are drawn once at thickness=1.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

PRIOR = "/data/disk1/home/ghe270/p3/prior_qualification"
DATA = os.path.join(PRIOR, "data", "reds_real_c9")
OUTS = os.path.join(PRIOR, "outputs", "reds_real_c9")
OUTS_REAL = os.path.join(PRIOR, "outputs", "realvsr")
M2B = "/data/disk1/home/ghe270/p3/m2b_rbv_ft"

CROP = 160
ZOOM = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# (clip, frame, x, y, description) -- identical to M1
PICKS = [
    ("016", 22, 932, 116, "foliage: fine leaf texture"),
    ("000", 24, 196, 20, "brick facade + railing, repeating pattern"),
    ("005", 24, 756, 100, "scaffold grid, thin structures"),
]

SUB = {
    "EDVR": (OUTS_REAL, "edvr_c9"),
    "RealBasicVSR": (OUTS, "gen_thres255_generator"),
    "OSEDiff": (OUTS_REAL, "osediff_c9"),
}


def to_u8(img):
    if img.dtype == np.uint8:
        return img
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)


def rd(p):
    a = cv2.imread(p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if a is not None else None


def label(img, text):
    # uint8 everywhere: vstack with a float64 tile promoted the canvas and
    # cv2.imwrite then truncated 0..1 to 0/1 (black tiles).
    img = to_u8(img)
    strip = np.full((22, img.shape[1], 3), 32, np.uint8)
    s = cv2.putText(strip, text, (4, 16), FONT, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return np.vstack([s, img])


def crop(img, x, y):
    h, w = img.shape[:2]
    x = max(0, min(x, w - CROP))
    y = max(0, min(y, h - CROP))
    c = img[y:y + CROP, x:x + CROP]
    return cv2.resize(c, (CROP * ZOOM, CROP * ZOOM),
                      interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", default="b1_dev")
    ap.add_argument("--b2", default="b2_dev")
    ap.add_argument("--out", default=os.path.join(M2B, "results",
                                                  "compare_textures_m2b.png"))
    ap.add_argument("--extra-clips", nargs="*", default=["016", "005"])
    args = ap.parse_args()

    rows = []
    for clip, fid, x, y, what in PICKS:
        fn = "%08d.png" % fid
        cells = []
        gt = rd(os.path.join(DATA, "GT", clip, fn))
        cells.append(label(crop(gt, x, y), "GT"))
        for name in ("EDVR", "RealBasicVSR", "OSEDiff"):
            root, sub = SUB[name]
            img = rd(os.path.join(root, sub, clip, fn))
            cells.append(label(crop(img, x, y) if img is not None
                               else np.full((CROP * ZOOM, CROP * ZOOM, 3), 40,
                                            np.uint8), name))
        for tag, arm in (("Ours_uniform", args.b1), ("Ours_uncertain", args.b2)):
            # an absolute arm path lets the full-scale /tmp outputs be used
            p = os.path.join(arm if os.path.isabs(arm) else
                             os.path.join(M2B, "outputs", arm), clip, fn)
            img = rd(p)
            if img is None:
                cells.append(label(np.full((CROP * ZOOM, CROP * ZOOM, 3), 40,
                                           np.uint8), tag + " N/A"))
                print("  MISSING %s/%s" % (arm, clip), flush=True)
            else:
                cells.append(label(crop(img, x, y), tag))
        rows.append(np.hstack(cells))
        print("row: %s frame %d  %s" % (clip, fid, what), flush=True)

    sheet = np.vstack(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(to_u8(sheet), cv2.COLOR_RGB2BGR))
    print("wrote", args.out, sheet.shape, flush=True)


if __name__ == "__main__":
    main()
