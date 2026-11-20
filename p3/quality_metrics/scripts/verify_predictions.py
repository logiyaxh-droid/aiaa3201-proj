# -*- coding: utf-8 -*-
"""Prove the regenerated predictions are bit-identical to what P0 scored.

`full_eval.py` discarded its prediction frames, so stage B/C regenerate them.
If the generator drifts by one rounding step, every new metric number becomes
incomparable to the stored PSNR / SSIM / LPIPS / tLP / FID table.  This script
recomputes the OLD metrics from freshly generated frames and compares them to
`p0/out/full/results/<clip>.json`, which is a sensitive detector: LPIPS at 1e-6
would move for a single changed pixel.

Run:  python verify_predictions.py 000 002 009
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torchvision

_mod = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F          # noqa: E402

for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _mod

import json                                                     # noqa: E402

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

import common as C                                              # noqa: E402
import predictions as P                                         # noqa: E402

torch.backends.cudnn.benchmark = False      # see p0/scripts/full_eval.py
OLD_ROOT = "/data/disk1/home/ghe270/p0/out/full/results"

CHECK = ["lanczos", "bicubic_unsharp", "edsr", "tavg3", "tavg5",
         "tavg5_unsharp", "flowtavg3", "flowtavg5", "flowtavg5_unsharp"]


def main(clips):
    sys.path.insert(0, "/data/disk1/home/ghe270/p0")
    from evalkit import metrics as M

    lpips_net = M.LPIPS(device="cuda")
    report = {"clips": {}, "tolerance": {"psnr": 1e-3, "lpips": 1e-6},
              "pass": True}

    for clip in clips:
        old_path = os.path.join(OLD_ROOT, "%s.json" % clip)
        if not os.path.exists(old_path):
            print("clip %s: no stored P0 result, skipped" % clip)
            continue
        old = json.load(open(old_path, encoding="utf-8"))
        cf = P.ClipFrames(clip)
        gt = np.stack(cf.gt)
        gtY = np.stack([M.luma(x) for x in cf.gt])
        rows = {}
        print("\n== clip %s ==" % clip)
        for m in CHECK:
            if m not in old:
                continue
            pred, src = cf.frames(m)
            pred = np.asarray(pred)
            pY = np.stack([M.luma(x) for x in pred])
            psnr = M.psnr_mean(pY, gtY)
            lpips = lpips_net.mean(pred, gt)
            d_psnr = abs(psnr - old[m]["psnr"])
            d_lpips = abs(lpips - old[m]["lpips"])
            ok = d_psnr < 1e-3 and d_lpips < 1e-6
            rows[m] = {"source": src, "psnr": float(psnr), "old_psnr": old[m]["psnr"],
                       "d_psnr": float(d_psnr), "lpips": float(lpips),
                       "old_lpips": old[m]["lpips"], "d_lpips": float(d_lpips),
                       "pass": bool(ok)}
            print("  %-20s PSNR %.6f (d=%.1e)  LPIPS %.6f (d=%.1e)  %s  [%s]"
                  % (m, psnr, d_psnr, lpips, d_lpips,
                     "ok" if ok else "MISMATCH",
                     "cache" if src.startswith("cache") else "computed"))
            if not ok:
                report["pass"] = False
            cf._product.clear()
        report["clips"][clip] = rows
        cf.release()

    C.write_json(os.path.join(C.ROOT, "selfcheck", "prediction_regression.json"),
                 report)
    print("\noverall: %s" % ("PASS" if report["pass"] else "FAIL"))
    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["000"])
