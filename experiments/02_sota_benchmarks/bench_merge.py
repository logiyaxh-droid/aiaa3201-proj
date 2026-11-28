#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge repeated ``bench_speed.py`` runs into one table, quoting the minimum.

Why this exists
---------------
The A6000 is shared.  Three identical runs of the same benchmark, hours apart,
disagreed by up to 6x on CPU methods (tavg3: 3.6 s, 4.1 s, 22.4 s) and by 3x on
EDSR (9.2 s, 14.7 s, 33.6 s).  The GPU's clock state and the other students' jobs
move on a scale of minutes, so a single run cannot be quoted.

``min`` over every repeat of every run is the statistic that needs only ONE clean
execution, so it is what the report quotes.  The per-run medians are also written
out: their spread is the honest error bar, and it is large enough that any
conclusion which flips between runs is flagged rather than reported.

Usage:
    /tmp/srenv/bin/python ~/p0/scripts/bench_merge.py \
        --runs ~/p0/out/bench/speed.json ~/p0/out/bench/speed_clean.json \
               ~/p0/out/bench/speed_final.json \
        --out ~/p0/out/bench/merged.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spot", default=None,
                    help="optional JSON from a spot re-check, merged as another run")
    args = ap.parse_args()

    runs = []
    for p in args.runs:
        p = os.path.expanduser(p)
        if os.path.exists(p):
            runs.append((os.path.basename(p), json.load(open(p, encoding="utf-8"))))
    if args.spot and os.path.exists(os.path.expanduser(args.spot)):
        runs.append((os.path.basename(os.path.expanduser(args.spot)),
                     json.load(open(os.path.expanduser(args.spot), encoding="utf-8"))))
    if not runs:
        raise SystemExit("no input runs found")

    out = {"runs": [r[0] for r in runs],
           "quoted_statistic": "min over all repeats of all runs",
           "why": "shared GPU; three identical runs disagreed by up to 6x on CPU methods "
                  "and 3x on EDSR, so min (one clean execution) is quoted and the "
                  "per-run medians are kept as the error bar",
           "methods": {}, "components": {}}

    names = set()
    for _, d in runs:
        names |= set(d.get("methods", {}))
    for m in sorted(names):
        per_run = []
        for rn, d in runs:
            v = d.get("methods", {}).get(m)
            if not v:
                continue
            per_run.append({"run": rn, "min_s": v["min_s"], "median_s": v["median_s"],
                            "max_s": v["max_s"],
                            "ms_per_frame": v.get("ms_per_frame"),
                            "peak_allocated_MiB": v.get("peak_allocated_MiB")})
        if not per_run:
            continue
        best = min(p["min_s"] for p in per_run)
        meds = [p["median_s"] for p in per_run]
        spread = max(meds) / max(min(meds), 1e-9)
        n = None
        for _, d in runs:
            if m in d.get("methods", {}):
                n = d.get("frames")
                break
        out["methods"][m] = {
            "min_s": best,
            "ms_per_frame": 1000.0 * best / n if n else None,
            "fps": n / best if n else None,
            "per_run_median_s": meds,
            "run_spread_ratio": spread,
            "unstable_between_runs": spread > 2.0,
            "peak_allocated_MiB": next((p["peak_allocated_MiB"] for p in per_run
                                        if p["peak_allocated_MiB"]), None),
            "per_run": per_run,
        }

    cnames = set()
    for _, d in runs:
        cnames |= set(d.get("components", {}))
    for c in sorted(cnames):
        vals = []
        for rn, d in runs:
            v = d.get("components", {}).get(c)
            if isinstance(v, dict) and "min_s" in v:
                vals.append((rn, v["min_s"], v["median_s"]))
        if not vals:
            continue
        out["components"][c] = {
            "min_s": min(v[1] for v in vals),
            "per_run_median_s": [v[2] for v in vals],
            "run_spread_ratio": max(v[2] for v in vals) / max(min(v[2] for v in vals), 1e-9),
            "per_run": [{"run": v[0], "min_s": v[1], "median_s": v[2]} for v in vals],
        }

    # reference run metadata
    d0 = runs[0][1]
    out["frames"] = d0.get("frames")
    out["lr_hw"] = d0.get("lr_hw")
    out["hr_hw"] = d0.get("hr_hw")
    out["device"] = d0.get("device")
    out["precision"] = d0.get("precision")

    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(args.out))), exist_ok=True)
    json.dump(out, open(os.path.expanduser(args.out), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    print("%-22s %9s %11s %8s %6s" % ("method", "min_s", "ms/frame", "fps", "spread"))
    for m, v in out["methods"].items():
        print("%-22s %9.2f %11.1f %8.2f %5.1fx %s"
              % (m, v["min_s"], v["ms_per_frame"] or 0, v["fps"] or 0,
                 v["run_spread_ratio"], "<- unstable" if v["unstable_between_runs"] else ""))
    print()
    print("%-30s %9s %6s" % ("component", "min_s", "spread"))
    for c, v in out["components"].items():
        print("%-30s %9.2f %5.1fx" % (c, v["min_s"], v["run_spread_ratio"]))
    print("->", args.out)


if __name__ == "__main__":
    main()
