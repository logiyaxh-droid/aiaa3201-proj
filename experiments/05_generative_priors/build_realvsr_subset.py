# -*- coding: utf-8 -*-
"""Build the RealVSR evaluation subset from the official test archives.

RealVSR ships `LQ_test.zip` and `GT_test.zip` with the *same* pixel grid
(512x1024): the LQ frame comes from a wide lens, the GT from a tele lens, then
both are aligned.  The official test script therefore uses an upscale=1 network
(EDVR_NoUp) and feeds LQ directly.

Our x4 checkpoints cannot consume a same-size pair, so this builder writes

    data/realvsr/GT/<clip>/<frame>.png     512x1024  (verbatim from GT_test.zip)
    data/realvsr/LQ4x/<clip>/<frame>.png   128x256   (LQ downsampled 4x, area)

and every x4 model reconstructs from LQ4x back onto the GT grid.  PIL's
BICUBIC is used for the downsample because it applies an anti-aliasing
prefilter when shrinking; cv2.INTER_CUBIC does not widen its kernel, so it
aliases and would make every model look worse for a reason unrelated to the
model.

Usage:
  python build_realvsr_subset.py --lq /path/LQ_test.zip --gt /path/GT_test.zip \
      --out ../data/realvsr --frames 20 21 22 23 24
"""
from __future__ import annotations

import argparse
import io
import os
import zipfile

from PIL import Image

SCALE = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lq", required=True, help="path to LQ_test.zip")
    ap.add_argument("--gt", required=True, help="path to GT_test.zip")
    ap.add_argument("--out", required=True, help="output root for GT/ and LQ4x/")
    ap.add_argument("--frames", type=int, nargs="+",
                    default=[20, 21, 22, 23, 24])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    zl = zipfile.ZipFile(args.lq)
    zg = zipfile.ZipFile(args.gt)
    clips = sorted({n.split("/")[1] for n in zl.namelist()
                    if n.count("/") == 2 and n.lower().endswith(".png")})
    if args.limit:
        clips = clips[:args.limit]
    print("clips", len(clips), "frames", args.frames)

    n_ok = 0
    for clip in clips:
        dg = os.path.join(args.out, "GT", clip)
        dl = os.path.join(args.out, "LQ4x", clip)
        os.makedirs(dg, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        for f in args.frames:
            nm = "%05d.png" % f
            try:
                b = zl.read("LQ_test/%s/%s" % (clip, nm))
                g = zg.read("GT_test/%s/%s" % (clip, nm))
            except KeyError:
                continue
            im = Image.open(io.BytesIO(b)).convert("RGB")
            w, h = im.size
            im.resize((w // SCALE, h // SCALE), Image.BICUBIC).save(
                os.path.join(dl, nm))
            with open(os.path.join(dg, nm), "wb") as fh:
                fh.write(g)          # GT stays byte-identical to the release
            n_ok += 1
    print("wrote %d frames" % n_ok)


if __name__ == "__main__":
    main()
