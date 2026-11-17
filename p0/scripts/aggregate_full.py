#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate ``out/full`` into the report tables (tLP + FID + deltas).

Reads every ``results/<clip>.json`` and ``feat/<method>/<clip>.npy`` written by
``scripts/full_eval.py`` and produces, in ``out/full/summary/``:

  table.json       per-method means over clips, plus per-method-per-clip rows
  table.md         the markdown table that goes into the report
  deltas.json      the paired comparisons the checklist asks for
  fid.md / fid.json
  perclip_*.csv    per-clip numbers, so a plot can be made without re-running

Every mean is over the *same 30 clips* for every method; a method missing a clip
is reported as such instead of being quietly averaged over fewer clips.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/p0"))

METHOD_ORDER = ["bicubic", "lanczos", "bicubic_unsharp", "edsr", "tavg3", "tavg5",
                "tavg5_unsharp", "flowtavg3", "flowtavg5", "flowtavg5_unsharp",
                "edvr", "tecogan"]
METRICS = ["psnr", "ssim", "lpips", "tlp", "flow_warp_lpips"]


def load_results(root):
    per = {}
    for p in sorted(glob.glob(os.path.join(root, "results", "*.json"))):
        clip = os.path.splitext(os.path.basename(p))[0]
        with open(p, "r", encoding="utf-8") as f:
            per[clip] = json.load(f)
    return per


def load_clips(root):
    """Expected clip list = the clip directories the run config points at."""
    cfg = json.load(open(os.path.join(root, "run_config.json"), encoding="utf-8"))
    gt = cfg["data"]["gt"]
    return sorted(d for d in os.listdir(gt) if os.path.isdir(os.path.join(gt, d)))


def fid_from_features(f1, f2):
    """Same number as scripts/fid_full.py; see there for why eigh, not scipy.sqrtm."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fid_full import fid as _fid
    return _fid(f1, f2)


def main():
    root = os.path.abspath(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                                             else "~/p0/out/full"))
    out_dir = os.path.join(root, "summary")
    os.makedirs(out_dir, exist_ok=True)

    per = load_results(root)
    clips = load_clips(root)
    methods = [m for m in METHOD_ORDER
               if any(m in per.get(c, {}) for c in clips)]

    missing = {m: [c for c in clips if m not in per.get(c, {})] for m in methods}
    complete = {m: [c for c in clips if m in per.get(c, {})] for m in methods}

    table, raw = {}, {}
    for m in methods:
        rows = {c: per[c][m] for c in complete[m]}
        entry = {}
        for k in METRICS:
            vals = [r[k] for r in rows.values() if r.get(k) is not None]
            entry[k] = float(np.mean(vals)) if vals else None
            entry[k + "_n"] = len(vals)
        entry["n_clips"] = len(rows)
        entry["missing_clips"] = missing[m]
        table[m] = entry
        raw[m] = {c: {k: rows[c].get(k) for k in METRICS + ["n_frames", "tlp_pairs"]}
                  for c in rows}

    # ---------------- FID (needs the full 30 clips for every method)
    fid = {}
    base = os.path.join(root, "feat", "gt")
    gt_feat = None
    if all(os.path.exists(os.path.join(base, "%s.npy" % c)) for c in clips):
        gt_feat = np.concatenate([np.load(os.path.join(base, "%s.npy" % c)) for c in clips])
    for m in methods:
        d = os.path.join(root, "feat", m)
        if gt_feat is None:
            break
        if not all(os.path.exists(os.path.join(d, "%s.npy" % c)) for c in clips):
            fid[m] = {"value": None, "n_frames": 0,
                      "reason": "features incomplete (%d/%d clips)"
                                % (len(glob.glob(os.path.join(d, "*.npy"))), len(clips))}
            continue
        f = np.concatenate([np.load(os.path.join(d, "%s.npy" % c)) for c in clips])
        fid[m] = {"value": fid_from_features(gt_feat, f),
                  "n_frames": int(f.shape[0]), "dims": int(f.shape[1])}
    if "gt" in fid:
        fid.pop("gt", None)

    json.dump({"per_method": table, "per_clip": raw, "clips": clips,
               "missing": missing},
              open(os.path.join(out_dir, "table.json"), "w", encoding="utf-8"),
              indent=1)
    json.dump(fid, open(os.path.join(out_dir, "fid.json"), "w", encoding="utf-8"), indent=1)

    # ---------------- deltas the checklist asks for
    def get(m, k):
        return table.get(m, {}).get(k)

    deltas = {
        "edsr_minus_bicubic": {k: (get("edsr", k) - get("bicubic", k))
                               for k in ("psnr", "ssim", "lpips", "tlp")},
        "tavg5_minus_bicubic": {k: (get("tavg5", k) - get("bicubic", k))
                                for k in ("psnr", "ssim", "lpips", "tlp")},
        "tavg3_minus_tavg5": {k: (get("tavg3", k) - get("tavg5", k))
                              for k in ("psnr", "tlp")},
        "flowtavg5_minus_tavg5": {k: (get("flowtavg5", k) - get("tavg5", k))
                                  for k in ("psnr", "ssim", "lpips", "tlp")},
        "bicubic_unsharp_minus_bicubic": {k: (get("bicubic_unsharp", k) - get("bicubic", k))
                                          for k in ("psnr", "ssim", "lpips", "tlp")},
        "flowtavg5_unsharp_minus_flowtavg5": {
            k: (get("flowtavg5_unsharp", k) - get("flowtavg5", k))
            for k in ("psnr", "ssim", "lpips", "tlp")},
        "edvr_minus_edsr": {k: (get("edvr", k) - get("edsr", k))
                            for k in ("psnr", "ssim", "lpips", "tlp")},
        "tecogan_minus_edvr": {k: (get("tecogan", k) - get("edvr", k))
                               for k in ("psnr", "ssim", "lpips", "tlp")},
        "tecogan_minus_bicubic": {k: (get("tecogan", k) - get("bicubic", k))
                                  for k in ("psnr", "ssim", "lpips", "tlp")},
    }
    # per-clip win counts for the claims that used to be "30/30"
    wins = {}
    for a, b in [("edsr", "bicubic"), ("flowtavg5", "tavg5"), ("edvr", "edsr"),
                 ("tecogan", "bicubic")]:
        common = [c for c in clips if c in complete[a] and c in complete[b]]
        for k, better in [("psnr", "high"), ("tlp", "low"), ("lpips", "low")]:
            xs = np.array([per[c][a][k] for c in common])
            ys = np.array([per[c][b][k] for c in common])
            wins["%s_%s_on_%s" % (a, b, k)] = {
                "n": len(common),
                "a_better": int((xs > ys).sum() if better == "high" else (xs < ys).sum()),
                "gains": {"min": float((xs - ys).min()), "median": float(np.median(xs - ys)),
                          "max": float((xs - ys).max())},
            }
    json.dump({"deltas": deltas, "per_clip_wins": wins},
              open(os.path.join(out_dir, "deltas.json"), "w", encoding="utf-8"), indent=1)

    # ---------------- markdown
    def f(v, nd=4):
        return "-" if v is None else ("%.*f" % (nd, v))

    lines = []
    lines.append("| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 正式 tLP ↓ | Flow-warp LPIPS ↓ | FID ↓ |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in methods:
        e = table[m]
        fv = fid.get(m, {}).get("value")
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            m, f(e["psnr"], 3), f(e["ssim"], 4), f(e["lpips"], 4), f(e["tlp"], 4),
            f(e["flow_warp_lpips"], 4), f(fv, 2)))
    md = "\n".join(lines) + "\n"
    open(os.path.join(out_dir, "table.md"), "w", encoding="utf-8").write(md)

    print(md)
    for m in methods:
        if table[m]["missing_clips"]:
            print("!! %s missing %d clips: %s"
                  % (m, len(table[m]["missing_clips"]), table[m]["missing_clips"]))
    print(json.dumps(deltas, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
