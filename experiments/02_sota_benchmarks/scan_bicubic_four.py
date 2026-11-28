#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0 task 10 -- which LR pre-shift maximises PSNR *through bicubic_four*?

``probe_grid.py`` measured the upsampler's grid exactly (ramp probe).  This
script turns that measurement into the number the old pipeline needed: sweep a
sub-pixel LR shift, push each shifted frame through the real
``lib/ops.py:bicubic_four`` in the TF1 environment, and report the PSNR curve
against GT.  The argmax is an *empirical* optimum, so it is also the honest way
to check the sign of the ramp-derived correction: if the ramp says "+0.376 LR px
cancels the offset" and this curve peaks near +0.375, the two agree by
independent routes.

Run in the TF1 env (it needs the real bicubic_four, and it is CPU-only):

  /tmp/tf1env/bin/python scripts/scan_bicubic_four.py \
      --clip 000 --frame 2 --out ~/p0/out/audit/bf_shift_scan.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/tmp/TecoGAN-master")


def main() -> int:
    import cv2
    import tensorflow as tf
    from lib.ops import bicubic_four

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/disk1/shared/aiaa3201/project1_video_sr")
    ap.add_argument("--clip", default="000")
    ap.add_argument("--frame", type=int, default=2)
    ap.add_argument("--lo", type=float, default=-0.75)
    ap.add_argument("--hi", type=float, default=0.75)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gt = cv2.imread(os.path.join(a.root, "val_sharp", a.clip, "%08d.png" % a.frame))
    lr = cv2.imread(os.path.join(a.root, "val_sharp_bicubic", a.clip, "%08d.png" % a.frame))
    if gt is None or lr is None:
        raise SystemExit("missing frames for clip %s frame %d" % (a.clip, a.frame))
    gtf = gt[:, :, ::-1].astype(np.float64) / 255.0   # BGR -> RGB, same space as bicubic_four

    ph = tf.placeholder(tf.float32, [1, None, None, 3])
    out = bicubic_four(ph)

    shifts = []
    n = int(round((a.hi - a.lo) / a.step))
    for k in range(n + 1):
        shifts.append(round(a.lo + k * a.step, 4))

    rows = {}
    with tf.Session() as sess:
        for s in shifts:
            M = np.float32([[1, 0, s], [0, 1, s]])
            lrs = cv2.warpAffine(lr, M, (lr.shape[1], lr.shape[0]), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
            x = lrs[:, :, ::-1].astype(np.float32) / 255.0       # BGR->RGB, keep float
            y = sess.run(out, {ph: x[None]})[0]
            y = np.clip(y, 0.0, 1.0)
            h = min(y.shape[0], gtf.shape[0])
            w = min(y.shape[1], gtf.shape[1])
            mse = float(((y[:h, :w] - gtf[:h, :w]) ** 2).mean())
            rows["%+.4f" % s] = {"psnr_rgb": float(10 * np.log10(1.0 / max(mse, 1e-12)))}

    best = max(rows.items(), key=lambda kv: kv[1]["psnr_rgb"])
    # parabola through the three points around the discrete optimum
    keys = sorted(rows, key=lambda k: float(k))
    bi = keys.index(best[0])
    sub = None
    if 0 < bi < len(keys) - 1:
        y0, y1, y2 = (rows[keys[bi - 1]]["psnr_rgb"], rows[best[0]]["psnr_rgb"],
                      rows[keys[bi + 1]]["psnr_rgb"])
        den = (y0 - 2 * y1 + y2)
        if abs(den) > 1e-9:
            sub = float(best[0]) + 0.5 * (y0 - y2) / den * a.step

    rep = {"clip": a.clip, "frame": a.frame, "impl": "lib/ops.py:bicubic_four (TF 1.15)",
           "shift_is_lr_px": True, "shifts": rows,
           "best_shift_grid": float(best[0]), "best_psnr_rgb": best[1]["psnr_rgb"],
           "best_shift_parabola": sub,
           "interpretation": "the LR pre-shift that makes bicubic_four best match GT; "
                             "compare with the ramp-derived correction from probe_grid.py"}
    txt = json.dumps(rep, indent=2)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
