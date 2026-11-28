# -*- coding: utf-8 -*-
"""Stage-A smoke test: do all three metrics materially work on clip 000?

This is not the self-check suite; it just proves the models load, accept our PNG
frames, and produce plausible numbers before we spend GPU time on the real
checks.  Prints per-metric timing and peak VRAM, which is what the roadmap asks
for in stage A step 4.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import common as C

CLIP = "000"
GT_DIR = os.path.join(C.GT_ROOT, CLIP)
BIC_DIR = os.path.join(C.BICUBIC_PRED, CLIP)
EDVR_DIR = os.path.join(C.EDVR_FRAMES, CLIP)
TECO_CLIP = os.path.join(C.TECO_FRAMES, CLIP)


def vram():
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**30, 2)


def timed(fn, *a, **kw):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = fn(*a, **kw)
    return out, time.time() - t0, vram()


def main():
    ids = C.clip_frame_ids(GT_DIR)
    print("clip %s: %d GT frames, size %s" % (CLIP, len(ids),
                                              C.read_rgb(os.path.join(GT_DIR, "00000000.png")).shape))
    gt = C.read_clip(GT_DIR, ids)
    bic = C.read_clip(BIC_DIR, ids)
    print("bicubic pred dir exists:", os.path.isdir(BIC_DIR),
          "| edvr cache:", os.path.isdir(EDVR_DIR),
          "| tecogan cache:", os.path.isdir(TECO_CLIP))

    print("\n== DISTS ==")
    dists = C.DistsMetric()
    s, dt, mem = timed(dists.per_frame, bic, gt)
    print("  bicubic: mean %.4f  median %.4f  (%.1fs, peak %.2f GiB)"
          % (s.mean(), np.median(s), dt, mem))

    print("\n== MUSIQ ==")
    musiq = C.MusiqMetric()
    s2, dt, mem = timed(musiq.per_frame, bic)
    print("  bicubic: mean %.4f  (%.1fs, peak %.2f GiB)" % (s2.mean(), dt, mem))
    s2g, dt, mem = timed(musiq.per_frame, gt)
    print("  GT     : mean %.4f  (%.1fs, peak %.2f GiB)" % (s2g.mean(), dt, mem))

    print("\n== DOVER ==")
    dover = C.DoverMetric()
    out, dt, mem = timed(dover.score_clip, BIC_DIR)
    print("  bicubic: technical %.4f  aesthetic %.4f  fused %.4f  (%.1fs, peak %.2f GiB)"
          % (out["technical"], out["aesthetic"], out["fused"], dt, mem))
    print("  sampled frame inds (technical, first 8):", out["frame_inds"]["technical"][:8])

    print("\n== DOVER repeatability (same seed) ==")
    out2, dt2, _ = timed(dover.score_clip, BIC_DIR)
    print("  technical %.4f vs %.4f  |  aesthetic %.4f vs %.4f  (%.1fs)"
          % (out["technical"], out2["technical"],
             out["aesthetic"], out2["aesthetic"], dt2))
    print("  identical:", out["technical"] == out2["technical"]
          and out["aesthetic"] == out2["aesthetic"])
    print("  frame inds identical:", out["frame_inds"] == out2["frame_inds"])


if __name__ == "__main__":
    main()
