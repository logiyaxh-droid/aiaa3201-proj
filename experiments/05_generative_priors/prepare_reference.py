# -*- coding: utf-8 -*-
"""Stage A: build the frozen reference outputs, regions and metrics.

Outputs (80 PNGs total):
  outputs/reference/edvr/<clip>/<frame>.png
  outputs/reference/ft/<clip>/<frame>.png
Results:
  results/reference/regions.json
  results/reference/edvr_metrics.json
  results/reference/ft_metrics.json
  results/reference/provenance.json
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np

import qa_common as Q

REF = os.path.join(ROOT, "outputs", "reference")
RES = os.path.join(ROOT, "results", "reference")


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    from pyiqa.archs.dists_arch import DISTS
    import common as C
    from evalkit import metrics as EM

    os.makedirs(RES, exist_ok=True)
    lp = EM.LPIPS(device="cuda", torch_home="/tmp/torch_home")
    dists = C.DistsMetric(device="cuda")
    edvr = Q.build_edvr(finetuned=False)
    ft = Q.build_edvr(finetuned=True)

    all_regions = {"clips": {}, "protocol": Q.record_provenance()["protocol"]}
    metrics = {"edvr": {"clips": {}}, "ft": {"clips": {}}}
    t0 = time.time()

    for clip in Q.DEV_CLIPS:
        ids = Q.dev_ids(clip)
        if len(ids) != 10:
            raise RuntimeError("%s: expected 10 dev frames, got %d" % (clip, len(ids)))
        gt = [Q.read_rgb(os.path.join(Q.GT_ROOT, clip, "%08d.png" % i)) for i in ids]
        lr = [Q.read_rgb(os.path.join(Q.LR_ROOT, clip, "%08d.png" % i)) for i in ids]

        base = Q.infer_clip(edvr, lr)
        flip = Q.infer_flip(edvr, lr)
        reg = Q.make_regions(clip, ids, gt, base, flip)
        all_regions["clips"][clip] = reg

        ft_pred = Q.infer_clip(ft, lr)
        for name, pred in (("edvr", base), ("ft", ft_pred)):
            for fid, arr in zip(ids, pred):
                Q.write_rgb(os.path.join(REF, name, clip, "%08d.png" % fid), arr)
            row = Q.score_frames(pred, gt, ids, reg, lp, dists)
            metrics[name]["clips"][clip] = row
            print("%s %s PSNR %.4f LPIPS %.5f texLPIPS %.5f uncLPIPS %.5f"
                  % (name, clip, row["psnr"], row["lpips"],
                     row["blocks"]["texture"]["lpips"],
                     row["blocks"]["uncertain"]["lpips"]), flush=True)

        dump(os.path.join(RES, "regions.json"), all_regions)
        dump(os.path.join(RES, "edvr_metrics.json"), metrics["edvr"])
        dump(os.path.join(RES, "ft_metrics.json"), metrics["ft"])

    for name in ("edvr", "ft"):
        metrics[name]["mean"] = Q.summarize(list(metrics[name]["clips"].values()))
        dump(os.path.join(RES, "%s_metrics.json" % name), metrics[name])
        print("%s mean %s" % (name, json.dumps(metrics[name]["mean"])), flush=True)

    dump(os.path.join(RES, "provenance.json"),
         Q.record_provenance({"stage": "A", "elapsed_s": time.time() - t0}))
    print("done in %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
