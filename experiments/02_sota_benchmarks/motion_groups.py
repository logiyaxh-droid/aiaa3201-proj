#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Task 6 -- motion-stratified analysis.

The point of stratification is to answer "does naive averaging fail *more* when
there is more motion, and where does flow alignment still fail?" -- questions a
single 30-clip mean cannot answer.

Grouping is FIXED BEFORE looking at any score, as the checklist requires:

  motion(clip) = median over all adjacent LR pairs of the per-pixel optical-flow
                 magnitude, computed on the **LR (bicubic) frames**, at 1/4 of LR
                 resolution with the same Farneback parameters used by the
                 alignment itself.
  Groups       = low / mid / high tertiles of that quantity over the 30 clips
                 (10 clips each).  Tertiles, not hand-picked thresholds, so the
                 split is reproducible and balanced.

Nothing on the model side (PSNR, LPIPS, tLP) enters the grouping.

Outputs ``out/motion/groups.json`` (the fixed clip->group assignment, so it can be
reused) and ``out/motion/analysis.json`` (per-group means + per-clip gains +
the flow-alignment failure list).

Usage:
    /tmp/srenv/bin/python ~/p0/scripts/motion_groups.py --out ~/p0/out/motion
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import cv2

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
LR_ROOT = B + "/val_sharp_bicubic"

# same estimator as the alignment code: 1/4-res Farneback, pyramid 3, win 15, iter 3
FB = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
          poly_sigma=1.2, flags=0)
SCALE = 4

# flow warp that disagrees by more than this many LR px still counts as a failure
OCCL_PX = 1.5


def flow_mag_stats(clip):
    files = sorted(glob.glob(os.path.join(LR_ROOT, clip, "*.png")))
    grays = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files]
    h, w = grays[0].shape
    hs, ws = h // SCALE, w // SCALE
    mags, resid_peak = [], []
    for t in range(len(grays) - 1):
        a = cv2.resize(grays[t], (ws, hs), interpolation=cv2.INTER_AREA)
        b = cv2.resize(grays[t + 1], (ws, hs), interpolation=cv2.INTER_AREA)
        f = cv2.calcOpticalFlowFarneback(a, b, None, **FB)
        mag = np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)
        mags.append(mag)
        # residual after removing the median flow = what is NOT global motion
        r = mag - np.median(mag)
        resid_peak.append(float(np.percentile(np.abs(r), 99)))
    m = np.stack(mags)
    return {
        "n_pairs": int(len(mags)),
        "median_mag_px": float(np.median(m)),           # grouping variable
        "p90_mag_px": float(np.percentile(m, 90)),
        "p99_resid_px": float(np.median(resid_peak)),    # non-global motion
        "mean_mag_px": float(m.mean()),
    }


def tertiles(clips, key):
    order = sorted(clips, key=lambda c: key[c])
    n = len(order)
    k = n // 3
    g = {}
    for i, c in enumerate(order):
        g[c] = "low" if i < k else ("mid" if i < 2 * k else "high")
    return g, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default="~/p0/out/full/results")
    ap.add_argument("--reuse", default=None, help="existing groups.json")
    args = ap.parse_args()
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out_dir, exist_ok=True)

    gpath = os.path.join(out_dir, "groups.json")
    if args.reuse and os.path.exists(args.reuse):
        gd = json.load(open(args.reuse, encoding="utf-8"))
        clips, stats, group = gd["clips"], gd["stats"], gd["group"]
        print("reused", args.reuse)
    else:
        clips = sorted(d for d in os.listdir(B + "/val_sharp")
                       if os.path.isdir(os.path.join(B + "/val_sharp", d)))
        stats = {}
        for c in clips:
            stats[c] = flow_mag_stats(c)
            print("%s  median %.2f px  p99resid %.2f px"
                  % (c, stats[c]["median_mag_px"], stats[c]["p99_resid_px"]), flush=True)
        group, order = tertiles(clips, {c: stats[c]["median_mag_px"] for c in clips})
        gd = {"clips": clips, "stats": stats, "group": group,
              "grouping_variable": "median per-pixel optical-flow magnitude on LR "
                                   "frames, all adjacent pairs, 1/4-res Farneback",
              "thresholds": {"low_max_px": max(stats[c]["median_mag_px"]
                                               for c in clips if group[c] == "low"),
                             "mid_max_px": max(stats[c]["median_mag_px"]
                                               for c in clips if group[c] == "mid")},
              "order": order}
        json.dump(gd, open(gpath, "w", encoding="utf-8"), indent=1)
        print("->", gpath)

    # ---- join with the full-run metrics
    res = {}
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(args.results), "*.json"))):
        res[os.path.splitext(os.path.basename(p))[0]] = json.load(open(p, encoding="utf-8"))

    methods = ["bicubic", "lanczos", "bicubic_unsharp", "edsr", "tavg3", "tavg5",
               "tavg5_unsharp", "flowtavg3", "flowtavg5", "flowtavg5_unsharp",
               "edvr", "tecogan"]
    methods = [m for m in methods if all(m in res.get(c, {}) for c in clips)]

    per_group = {}
    for g in ("low", "mid", "high"):
        cs = [c for c in clips if group[c] == g]
        per_group[g] = {"n_clips": len(cs), "clips": cs,
                        "median_motion_px": float(np.median(
                            [stats[c]["median_mag_px"] for c in cs])),
                        "methods": {}}
        for m in methods:
            per_group[g]["methods"][m] = {
                k: float(np.mean([res[c][m][k] for c in cs]))
                for k in ("psnr", "ssim", "lpips", "tlp", "flow_warp_lpips")}

    # ---- the two questions the checklist asks
    gains = {}
    for c in clips:
        gains[c] = {
            "tavg5_minus_bicubic_psnr": res[c]["tavg5"]["psnr"] - res[c]["bicubic"]["psnr"],
            "tavg5_minus_bicubic_tlp": res[c]["tavg5"]["tlp"] - res[c]["bicubic"]["tlp"],
            "flowtavg5_minus_tavg5_psnr": (res[c]["flowtavg5"]["psnr"]
                                           - res[c]["tavg5"]["psnr"]),
            "flowtavg5_minus_bicubic_psnr": (res[c]["flowtavg5"]["psnr"]
                                             - res[c]["bicubic"]["psnr"]),
            "flowtavg5_minus_bicubic_tlp": (res[c]["flowtavg5"]["tlp"]
                                            - res[c]["bicubic"]["tlp"]),
            "flowtavg5_minus_tavg5_tlp": (res[c]["flowtavg5"]["tlp"]
                                          - res[c]["tavg5"]["tlp"]),
            "edvr_minus_edsr_psnr": res[c]["edvr"]["psnr"] - res[c]["edsr"]["psnr"],
            "motion": stats[c]["median_mag_px"],
            "resid": stats[c]["p99_resid_px"],
            "group": group[c],
        }

    summary = {}
    for key in gains[clips[0]]:
        if key in ("group",):
            continue
        for g in ("low", "mid", "high"):
            cs = [c for c in clips if group[c] == g]
            summary.setdefault(key, {})[g] = float(np.mean([gains[c][key] for c in cs]))

    # correlation of each gain with motion (rank-based, n=30)
    from scipy.stats import spearmanr
    corr = {}
    mv = np.array([stats[c]["median_mag_px"] for c in clips])
    for key in gains[clips[0]]:
        if key in ("group", "motion", "resid"):
            continue
        gv = np.array([gains[c][key] for c in clips])
        rho, p = spearmanr(mv, gv)
        corr[key] = {"spearman_rho": float(rho), "p_value": float(p), "n": len(clips)}

    # ---- where does flow alignment still fail, and where does it hurt
    fails = sorted(clips, key=lambda c: gains[c]["flowtavg5_minus_bicubic_psnr"])
    worst = [{"clip": c, "group": group[c],
              "motion_px": stats[c]["median_mag_px"],
              "resid_px": stats[c]["p99_resid_px"],
              "flowtavg5_minus_bicubic_psnr": gains[c]["flowtavg5_minus_bicubic_psnr"],
              "flowtavg5_minus_tavg5_psnr": gains[c]["flowtavg5_minus_tavg5_psnr"]}
             for c in fails[:5]]

    out = {"grouping": {"variable": gd["grouping_variable"],
                        "thresholds": gd.get("thresholds"),
                        "groups": {g: [c for c in clips if group[c] == g]
                                   for g in ("low", "mid", "high")}},
           "per_clip": {"stats": stats, "group": group, "gains": gains},
           "per_group": per_group,
           "group_means": summary,
           "motion_correlation": corr,
           "flow_align_worst_clips": worst,
           "n_methods": len(methods),
           "methods": methods}
    json.dump(out, open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8"),
              indent=1)

    print("\n== per-group means ==")
    for m in methods:
        row = [per_group[g]["methods"][m] for g in ("low", "mid", "high")]
        print("%-20s psnr %6.2f/%6.2f/%6.2f   tLP %.4f/%.4f/%.4f"
              % (m, row[0]["psnr"], row[1]["psnr"], row[2]["psnr"],
                 row[0]["tlp"], row[1]["tlp"], row[2]["tlp"]))
    print("\n== gain vs motion (spearman) ==")
    for k, v in corr.items():
        print("%-34s rho %+.3f  p %.3f" % (k, v["spearman_rho"], v["p_value"]))
    print("\nworst flow-aligned clips vs bicubic:")
    for w in worst:
        print("  %s [%s] motion %.2f resid %.2f  dPSNR(flow-bic) %+.3f"
              % (w["clip"], w["group"], w["motion_px"], w["resid_px"],
                 w["flowtavg5_minus_bicubic_psnr"]))
    print("->", os.path.join(out_dir, "analysis.json"))


if __name__ == "__main__":
    main()
