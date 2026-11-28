# -*- coding: utf-8 -*-
"""Score a candidate on a frame subset using the frozen region definitions.

Used for DOVE, which needs (F-1) % 8 == 0 and therefore ran on frames 40-48
(9 frames) instead of the 10-frame reference set.  This recomputes the EDVR and
fine-tuned EDVR references on exactly the same 9 frames so the comparison stays
paired, then scores the candidate.

Usage:
  python score_subset.py --name dove9 --dir outputs/dove9_720 --frames 40 49
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import torch

import qa_common as Q

REF_RES = os.path.join(ROOT, "results", "reference")
REF_OUT = os.path.join(ROOT, "outputs", "reference")


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
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--end", type=int, default=49)
    args = ap.parse_args()

    import common as C
    from evalkit import metrics as EM
    from evalkit import temporal as ET

    lp = EM.LPIPS(device="cuda", torch_home="/tmp/torch_home")
    dists = C.DistsMetric(device="cuda")
    regions = json.load(open(os.path.join(REF_RES, "regions.json"),
                             encoding="utf-8"))["clips"]
    edvr = Q.build_edvr(finetuned=False)
    ft = Q.build_edvr(finetuned=True)

    out = {"name": args.name, "frames": [args.start, args.end - 1],
           "clips": {}, "reference": {}}
    for clip in Q.DEV_CLIPS:
        ids = [i for i in Q.dev_ids(clip) if args.start <= i < args.end]
        gt = [Q.read_rgb(os.path.join(Q.GT_ROOT, clip, "%08d.png" % i)) for i in ids]
        lr = [Q.read_rgb(os.path.join(Q.LR_ROOT, clip, "%08d.png" % i)) for i in ids]

        refs = {"edvr": Q.infer_clip(edvr, lr), "ft": Q.infer_clip(ft, lr)}
        pred = []
        for i in ids:
            p = os.path.join(args.dir, clip, "%03d.png" % (i - args.start))
            if not os.path.exists(p):
                p = os.path.join(args.dir, clip, "%08d.png" % i)
            if not os.path.exists(p):
                raise SystemExit("missing candidate frame: %s" % p)
            pred.append(Q.read_rgb(p))

        for tag, frames in (("edvr", refs["edvr"]), ("ft", refs["ft"]),
                            ("cand", pred)):
            row = Q.score_frames(frames, gt, ids, regions[clip], lp, dists)
            pairs = ET.pairs_of(ids, 0)
            if pairs:
                d_sr = lp.pair(np.stack([frames[a] for a, _ in pairs]),
                               np.stack([frames[b] for _, b in pairs]))
                d_gt = lp.pair(np.stack([gt[a] for a, _ in pairs]),
                               np.stack([gt[b] for _, b in pairs]))
                row["tlp"] = float(np.abs(d_sr - d_gt).mean())
            else:
                row["tlp"] = None
            out["clips"].setdefault(clip, {})[tag] = row
            if tag != "cand":
                out["reference"].setdefault(tag, {})[clip] = row
            print("%s %s %s PSNR %.4f LPIPS %.5f tex %.5f unc %.5f tLP %s"
                  % (args.name, clip, tag, row["psnr"], row["lpips"],
                     row["blocks"]["texture"]["lpips"],
                     row["blocks"]["uncertain"]["lpips"],
                     ("%.5f" % row["tlp"]) if row["tlp"] is not None else "N/A"),
                  flush=True)
        dump(os.path.join(ROOT, "results", args.name, "metrics_subset.json"), out)

    # Aggregate candidate and references over the same frames.
    agg = {}
    for tag in ("edvr", "ft", "cand"):
        rows = [out["clips"][c][tag] for c in out["clips"]]
        agg[tag] = Q.summarize(rows)
        agg[tag]["tlp"] = float(np.mean([r["tlp"] for r in rows
                                         if r["tlp"] is not None]))
    out["mean"] = agg
    out["compare"] = {}
    for ref in ("edvr", "ft"):
        cmp = {}
        for k in ("psnr", "lpips", "dists", "texture_lpips", "texture_dists",
                  "uncertain_lpips", "uncertain_dists", "flat_lpips", "tlp"):
            c, b = agg["cand"][k], agg[ref][k]
            cmp[k] = {"candidate": c, "reference": b, "delta": c - b,
                      "rel_pct": (100.0 * (c - b) / b) if b else None}
        out["compare"][ref] = cmp
    dump(os.path.join(ROOT, "results", args.name, "metrics_subset.json"), out)
    print("\ncand  ", json.dumps(agg["cand"]), flush=True)
    print("ft    ", json.dumps(agg["ft"]), flush=True)
    print("texture rel vs ft %.2f%%, uncertain rel vs ft %.2f%%"
          % (out["compare"]["ft"]["texture_lpips"]["rel_pct"],
             out["compare"]["ft"]["uncertain_lpips"]["rel_pct"]), flush=True)


if __name__ == "__main__":
    main()
