# -*- coding: utf-8 -*-
"""Score a VideoLQ run with locally-available metrics only.

`score_vlq.py` calls `pyiqa.create_metric("niqe")`, which reaches the network to
fetch weights and dies with "Network is unreachable" on this host.  MUSIQ is
already present as a local checkpoint, so reuse `quality_metrics/scripts/common.py`
(which loads it from an explicit path) and skip NIQE rather than silently
reporting a zero.

Usage:
  python score_vlq_local.py --name edvr_v2 --dir outputs/vlq/edvr_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3 = "/data/disk1/home/ghe270/p3"
sys.path.insert(0, os.path.join(P3, "quality_metrics", "scripts"))
os.environ.setdefault("TORCH_HOME",
                      os.path.join(P3, "quality_metrics", "weights", "torch_home"))

SRC = os.path.join(ROOT, "data", "vlq_eval")
OUT = os.path.join(ROOT, "results", "vlq")


def read_rgb(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    import common as C

    musiq = C.MusiqMetric()
    clips = sorted(d for d in os.listdir(SRC)
                   if os.path.isdir(os.path.join(SRC, d)))
    rows, miss = {}, []
    for clip in clips:
        sd = os.path.join(SRC, clip)
        od = os.path.join(args.dir, clip)
        ids = sorted(f for f in os.listdir(sd) if f.endswith(".png"))
        preds, refs = [], []
        for f in ids:
            p = os.path.join(od, f)
            if not os.path.exists(p):
                miss.append("%s/%s" % (clip, f))
                continue
            preds.append(read_rgb(p))
            refs.append(read_rgb(os.path.join(sd, f)))
        if not preds:
            continue
        shapes = {x.shape for x in preds}
        if len(shapes) != 1:
            raise SystemExit("%s: inconsistent shapes %s" % (clip, shapes))
        mq = float(musiq.per_frame(preds).mean())
        x = np.stack(preds).astype(np.float64)
        r = np.stack(refs).astype(np.float64)
        mse = float(((x - r) ** 2).mean())
        rows[clip] = {"musiq": mq, "n": len(preds),
                      "size": list(preds[0].shape[:2]),
                      "self_psnr": float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))}
        print("%s musiq %.3f selfPSNR %.3f (%d frames)"
              % (clip, mq, rows[clip]["self_psnr"], len(preds)), flush=True)

    if not rows:
        raise SystemExit("no comparable frames under %s" % args.dir)

    mean = {k: float(np.mean([v[k] for v in rows.values()]))
            for k in ("musiq", "self_psnr")}
    out = {"method": args.name, "dir": os.path.abspath(args.dir),
           "n_clips": len(rows), "n_frames": sum(v["n"] for v in rows.values()),
           "missing": miss, "musiq": mean["musiq"],
           "self_psnr": mean["self_psnr"],
           "niqe": None,
           "note": "NIQE skipped: pyiqa.create_metric('niqe') fetches weights over "
                   "the network and this host has no egress. MUSIQ uses the local "
                   "checkpoint. self_psnr is agreement with the native frame, not "
                   "accuracy against truth (VideoLQ has no GT)."}
    dump(os.path.join(OUT, "%s.json" % args.name), out)
    print("\n%s MEAN musiq %.3f selfPSNR %.3f"
          % (args.name, out["musiq"], out["self_psnr"]), flush=True)


if __name__ == "__main__":
    main()
