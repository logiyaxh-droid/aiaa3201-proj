# -*- coding: utf-8 -*-
"""Final validation of the selected configuration on all 30 REDS val clips.

The three-clip diagnostic sweep is a screening step; the acceptance criteria in
the plan require the full 30 clips before any claim is made.  This script reuses
the exact evaluation path (`evaluate.py`) used for the baseline, so the comparison
against P0's stored EDVR row is apples to apples.

Outputs:
  results/<name>/metrics_30.json   per-clip and aggregate
  results/<name>/compare_30.md     table against the stored baseline

It also reports the paired per-clip wins/losses, because an aggregate win driven
by a handful of clips is a different claim from a consistent improvement.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = "/data/disk1/home/ghe270/p3/quality_metrics"
PY = os.path.join(ROOT, "venv", "bin", "python")
EVAL = os.path.join(ROOT, "fine_tune", "scripts", "evaluate.py")
LOGS = os.path.join(ROOT, "fine_tune", "logs")
RESULTS = os.path.join(ROOT, "fine_tune", "results")
BASELINE = "/data/disk1/home/ghe270/p0/out/full/results"


def baseline_30():
    clips = sorted(f[:-5] for f in os.listdir(BASELINE) if f.endswith(".json"))
    out = {}
    for c in clips:
        path = os.path.join(BASELINE, c + ".json")
        if not os.path.isfile(path):
            continue
        d = json.load(open(path, encoding="utf-8"))
        if "edvr" in d:
            out[c] = {k: d["edvr"][k] for k in ("psnr", "lpips", "tlp")}
    # DISTS per clip from the stage-C results
    for c in list(out):
        p = os.path.join(ROOT, "results", "reds30", "dists", "%s.json" % c)
        if os.path.exists(p):
            dd = json.load(open(p, encoding="utf-8"))
            if "edvr" in dd and dd["edvr"].get("status") == "ok":
                out[c]["dists"] = dd["edvr"]["score"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    clips = sorted(baseline_30())
    out = os.path.join(RESULTS, args.name, "metrics_30.json")
    if not os.path.exists(out):
        with open(os.path.join(LOGS, "final30_%s.log" % args.name), "w") as f:
            rc = subprocess.run([PY, EVAL, "--checkpoint", args.checkpoint,
                                 "--name", args.name, "--clips", *clips],
                                stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise SystemExit("evaluation failed, see logs")

    got = json.load(open(os.path.join(RESULTS, args.name, "metrics.json"),
                         encoding="utf-8"))
    base = baseline_30()
    clips_done = sorted(got["clips"])
    missing = [c for c in base if c not in clips_done]

    rows, wins = [], {"psnr": 0, "lpips": 0, "dists": 0, "tlp": 0}
    for c in clips_done:
        if c not in base:
            continue
        g, b = got["clips"][c], base[c]
        r = {"clip": c}
        for k in ("psnr", "lpips", "dists", "tlp"):
            if k in b:
                r[k] = g[k]
                r["base_" + k] = b[k]
                r["d_" + k] = g[k] - b[k]
                better = (r["d_" + k] > 0) if k == "psnr" else (r["d_" + k] < 0)
                wins[k] += int(better)
        rows.append(r)

    agg = {}
    for k in ("psnr", "lpips", "dists", "tlp"):
        vals = [r[k] for r in rows if k in r]
        bvals = [r["base_" + k] for r in rows if "base_" + k in r]
        if vals:
            agg[k] = {"mean": sum(vals) / len(vals),
                      "base_mean": sum(bvals) / len(bvals),
                      "delta": sum(vals) / len(vals) - sum(bvals) / len(bvals),
                      "n": len(vals)}

    report = {"name": args.name, "checkpoint": args.checkpoint,
              "n_clips": len(rows), "missing": missing,
              "aggregate": agg, "paired_wins": wins, "per_clip": rows}
    json.dump(report, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    md = ["# 30 片段最终验证：%s" % args.name, "",
          "对照为 P0 存储的 EDVR 行（同 clip、同口径）。", "",
          "| 指标 | 基线 | 本配置 | Δ | 逐片段更优数 |", "|---|---|---|---|---|"]
    for k, label in (("psnr", "PSNR ↑"), ("lpips", "LPIPS ↓"),
                     ("dists", "DISTS ↓"), ("tlp", "tLP ↓")):
        a = agg.get(k)
        if a:
            md.append("| %s | %.4f | %.4f | %+.4f | %d/%d |"
                      % (label, a["base_mean"], a["mean"], a["delta"],
                         wins[k], a["n"]))
    md += ["", "缺片段：%s" % (missing if missing else "无"), ""]
    open(os.path.join(RESULTS, args.name, "compare_30.md"), "w",
         encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
