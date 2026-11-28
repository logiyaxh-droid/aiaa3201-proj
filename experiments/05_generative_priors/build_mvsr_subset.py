# -*- coding: utf-8 -*-
"""Build the MVSR4x evaluation subset.

MVSR4x is captured with a phone's dual-view camera: an ultra-wide lens (13mm
equivalent) gives the low-resolution frame, a telephoto lens (90mm equivalent)
gives the high-resolution frame, and both run at the same time.  The captures
are already aligned and saved on the *same* pixel grid, so the scale factor is
not visible from the file sizes; it has to be measured.

Bandwidth measurement (high-frequency energy above 0.25 of Nyquist, percent of
total spectral energy, clip 003 / 065 frame 40)::

    tele native        0.078 / 0.030
    wide native        0.033 / 0.004
    tele down-2, up-2  0.059 / 0.023
    tele down-4, up-4  0.029 / 0.009    <- matches wide

so MVSR4x really is the x4 pair it claims to be (unlike RealVSR, whose LQ and GT
differ by about 2x only).

Layout after extraction::

    MVSR4x/tele_test/<clip>/NNNNN.png    HR  (telephoto), 1024x1024
    MVSR4x/wide_test/<clip>/NNNNN.png    LR  (ultra-wide), 1024x1024

This builder writes::

    data/mvsr4x/GT/<clip>/<id>.png      tele frame, native size
    data/mvsr4x/LQ4x/<clip>/<id>.png    wide frame downsampled 4x (area)

The wide frame is downsampled rather than used as-is because the two lenses are
aligned onto one grid: feeding the 1024x1024 wide frame to an x4 model would
hand it an input that is already at target resolution.

Usage:
  python build_mvsr_subset.py --root /path/to/MVSR4x --out ../data/mvsr4x \
      --frames 40 41 42 43 44
"""
from __future__ import annotations

import argparse
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing tele_test/ and wide_test/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tele_root = os.path.join(args.root, "tele_test")
    wide_root = os.path.join(args.root, "wide_test")
    clips = sorted(d for d in os.listdir(tele_root)
                   if os.path.isdir(os.path.join(tele_root, d))
                   and os.path.isdir(os.path.join(wide_root, d)))
    if args.limit:
        clips = clips[:args.limit]
    print("paired clips", len(clips))

    n_ok = 0
    for clip in clips:
        dg = os.path.join(args.out, "GT", clip)
        dl = os.path.join(args.out, "LQ4x", clip)
        os.makedirs(dg, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        for f in args.frames:
            nm = "%05d.png" % f
            hp = os.path.join(tele_root, clip, nm)
            lp = os.path.join(wide_root, clip, nm)
            if not (os.path.exists(hp) and os.path.exists(lp)):
                continue
            hr = Image.open(hp).convert("RGB")
            lr = Image.open(lp).convert("RGB")
            hr.save(os.path.join(dg, nm))
            tgt = (hr.width // 4, hr.height // 4)
            # PIL's BICUBIC widens its kernel when shrinking, so it acts as an
            # anti-aliased prefilter.  cv2.INTER_CUBIC does not, and aliases;
            # that aliasing was measured to cost ~1.5 dB on the REDS val set and
            # would make every model look worse for a reason unrelated to the
            # model.  cv2.INTER_AREA would be equally valid here.
            lr.resize(tgt, Image.BICUBIC).save(os.path.join(dl, nm))
            n_ok += 1
        print("%s gt=%s lq4x=%s" % (clip, hr.size, tgt), flush=True)
    print("wrote %d frames" % n_ok)


if __name__ == "__main__":
    main()
