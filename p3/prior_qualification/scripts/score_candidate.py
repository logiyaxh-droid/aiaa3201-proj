# -*- coding: utf-8 -*-
"""Score one candidate's exported PNG frames against the frozen references.

Candidate layout: <candidate_dir>/<clip>/<frame>.png at 1280x720 RGB.
All scores use exactly the same ids and block coordinates as the references.

Usage:
  python score_candidate.py --name osediff --dir outputs/osediff
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np

import qa_common as Q

REF_RES = os.path.join(ROOT, "results", "reference")
REF_OUT = os.path.join(ROOT, "outputs", "reference")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def tlp_for(pred, gt, ids, lp):
    from evalkit import temporal as ET

    pairs = ET.pairs_of(ids, 0)
    if len(pairs) < 5:
        return None, len(pairs)
    d_sr = lp.pair(np.stack([pred[a] for a, _ in pairs]),
                   np.stack([pred[b] for _, b in pairs]))
    d_gt = lp.pair(np.stack([gt[a] for a, _ in pairs]),
                   np.stack([gt[b] for _, b in pairs]))
    return float(np.abs(d_sr - d_gt).mean()), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    from pyiqa.archs.dists_arch import DISTS
    import common as C
    from evalkit import metrics as EM

    lp = EM.LPIPS(device="cuda", torch_home="/tmp/torch_home")
    dists = C.DistsMetric(device="cuda")
    regions = load_json(os.path.join(REF_RES, "regions.json"))["clips"]
    refs = {n: load_json(os.path.join(REF_RES, "%s_metrics.json" % n))
            for n in ("edvr", "ft")}

    out = {"name": args.name, "dir": os.path.abspath(args.dir),
           "clips": {}, "missing": []}
    for clip in Q.DEV_CLIPS:
        ids = Q.dev_ids(clip)
        gt, pred = [], []
        for fid in ids:
            p = os.path.join(args.dir, clip, "%08d.png" % fid)
            if not os.path.exists(p):
                out["missing"].append("%s/%08d" % (clip, fid))
                continue
            gt.append(Q.read_rgb(os.path.join(Q.GT_ROOT, clip, "%08d.png" % fid)))
            pred.append(Q.read_rgb(p))
        if not pred:
            continue
        used = ids[:len(pred)]
        row = Q.score_frames(pred, gt, used, regions[clip], lp, dists)
        row["tlp"], row["n_pairs"] = tlp_for(pred, gt, used, lp)
        out["clips"][clip] = row
        print("%s %s PSNR %.3f LPIPS %.5f tex %.5f unc %.5f tLP %s"
              % (args.name, clip, row["psnr"], row["lpips"],
                 row["blocks"]["texture"]["lpips"],
                 row["blocks"]["uncertain"]["lpips"],
                 row["tlp"]), flush=True)

    if not out["clips"]:
        raise SystemExit("no candidate frames found under %s" % args.dir)
    if out["missing"] and not args.allow_missing:
        raise SystemExit("missing frames: %s" % out["missing"][:10])

    out["mean"] = Q.summarize(list(out["clips"].values()))
    out["compare"] = {}
    for ref in ("edvr", "ft"):
        rm = refs[ref]["mean"]
        cmp = {}
        for key in ("psnr", "lpips", "dists",
                    "texture_lpips", "texture_dists",
                    "uncertain_lpips", "uncertain_dists",
                    "flat_lpips"):
            c = out["mean"][key]
            b = rm[key]
            cmp[key] = {"candidate": c, "reference": b,
                        "delta": c - b,
                        "rel_pct": (100.0 * (c - b) / b) if b else None}
        out["compare"][ref] = cmp
    out["per_clip_texture_delta_vs_ft"] = {
        c: out["clips"][c]["blocks"]["texture"]["lpips"] -
           refs["ft"]["clips"][c]["blocks"]["texture"]["lpips"]
        for c in out["clips"] if c in refs["ft"]["clips"]}
    out["per_clip_uncertain_delta_vs_ft"] = {
        c: out["clips"][c]["blocks"]["uncertain"]["lpips"] -
           refs["ft"]["clips"][c]["blocks"]["uncertain"]["lpips"]
        for c in out["clips"] if c in refs["ft"]["clips"]}
    dump(os.path.join(ROOT, "results", args.name, "metrics.json"), out)
    print("mean", json.dumps(out["mean"]), flush=True)
    print("texture rel vs ft %.2f%%, uncertain rel %.2f%%"
          % (out["compare"]["ft"]["texture_lpips"]["rel_pct"],
             out["compare"]["ft"]["uncertain_lpips"]["rel_pct"]), flush=True)


if __name__ == "__main__":
    main()
