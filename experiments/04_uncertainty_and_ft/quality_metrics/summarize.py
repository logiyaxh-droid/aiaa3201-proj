# -*- coding: utf-8 -*-
"""Build the tables and figures from the raw per-(clip, method) JSONs.

Every table in the report is generated here.  Nothing is hand-edited: if a
number appears in `summary/`, this script put it there from `results/`.

What it produces
----------------
summary/<scope>/metrics_table.csv      method x metric, clip-level mean + std
summary/<scope>/metrics_table.md       the same, for reading
summary/<scope>/per_clip.csv           every clip x method x metric value
summary/<scope>/deltas.csv             paired differences vs bicubic and vs edvr
summary/<scope>/rank_changes.md        does DISTS/MUSIQ/DOVER reorder the methods?
summary/<scope>/agreement.csv          Spearman between metrics across clips

Design rules taken from the roadmap
-----------------------------------
* The main number is the mean over CLIP-level values, and the clip-to-clip
  standard deviation is reported next to it.  A pooled 3000-frame mean is not
  the headline number and is not computed here.
* Coverage is explicit: a missing (clip, method, metric) cell is reported as
  missing and `coverage` in the output says how many of the expected cells were
  filled.  Absent values are never written as 0.
* DISTS is lower-better, MUSIQ / DOVER higher-better; the direction is carried
  in the column metadata so a ranking cannot silently invert.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C

METHOD_ORDER = ["bicubic", "lanczos", "bicubic_unsharp", "edsr", "tavg3", "tavg5",
                "tavg5_unsharp", "flowtavg3", "flowtavg5", "flowtavg5_unsharp",
                "edvr", "tecogan"]

DIRECTION = {"dists": -1, "musiq": +1, "dover_technical": +1, "dover_aesthetic": +1}
# -1 = lower is better

METRIC_COLUMNS = ["dists", "musiq", "dover_technical", "dover_aesthetic"]


def load_scope(scope):
    """{metric: {clip: {method: payload}}} for the ok rows only."""
    root = os.path.join(C.ROOT, "results", scope)
    out = {}
    for metric in ("dists", "musiq", "dover"):
        d = {}
        for path in sorted(glob.glob(os.path.join(root, metric, "*.json"))):
            clip = os.path.splitext(os.path.basename(path))[0]
            rows = json.load(open(path, encoding="utf-8"))
            d[clip] = {m: v for m, v in rows.items() if v.get("status") == "ok"}
        if d:
            out[metric] = d
    return out


def collect(scope):
    """-> {(clip, method): {metric_column: value}} plus the expected cell count."""
    data = load_scope(scope)
    cells = {}
    for clip, methods in data.get("dists", {}).items():
        for m, v in methods.items():
            cells.setdefault((clip, m), {})["dists"] = v["score"]
    for clip, methods in data.get("musiq", {}).items():
        for m, v in methods.items():
            cells.setdefault((clip, m), {})["musiq"] = v["score"]
    for clip, methods in data.get("dover", {}).items():
        for m, v in methods.items():
            cells.setdefault((clip, m), {})["dover_technical"] = v["technical"]
            cells.setdefault((clip, m), {})["dover_aesthetic"] = v["aesthetic"]

    # what was expected: whatever the run_config says, times the metrics asked
    cfg_path = os.path.join(C.ROOT, "results", scope, "run_config.json")
    expected = {"clips": [], "methods": [], "metrics": []}
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        expected = {"clips": cfg.get("clips", []),
                    "methods": cfg.get("methods", []),
                    "metrics": cfg.get("metrics", [])}
    return cells, expected


def clip_level(scope):
    cells, expected = collect(scope)
    clips = sorted({c for c, _ in cells})
    methods = [m for m in METHOD_ORDER if any(mm == m for _, mm in cells)]
    if not methods:
        methods = sorted({m for _, m in cells})

    per_clip = []
    for (clip, m), vals in sorted(cells.items()):
        row = {"clip": clip, "method": m}
        row.update({k: vals.get(k) for k in METRIC_COLUMNS})
        per_clip.append(row)

    table = {}
    for m in methods:
        row = {}
        for k in METRIC_COLUMNS:
            v = [cells[(c, m)][k] for c in clips
                 if (c, m) in cells and k in cells[(c, m)]]
            if v:
                row[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))
                          if len(v) > 1 else 0.0, "n": len(v),
                          "min": float(np.min(v)), "max": float(np.max(v))}
            else:
                row[k] = None
        table[m] = row

    coverage = {"clips": len(clips), "methods": len(methods),
                "expected_cells": len(clips) * len(methods) * len(METRIC_COLUMNS),
                "filled_cells": sum(len(v) for v in cells.values()),
                "expected_from_config": {
                    "clips": len(expected.get("clips") or []),
                    "methods": len(expected.get("methods") or []),
                    "metrics": expected.get("metrics"),
                }}
    return per_clip, table, coverage, clips, methods


def write_table_csv(table, methods, path):
    lines = ["method," + ",".join(
        "%s_mean,%s_std,%s_n" % (k, k, k) for k in METRIC_COLUMNS)]
    for m in methods:
        cells = []
        for k in METRIC_COLUMNS:
            v = table[m][k]
            if v is None:
                cells += ["", "", "0"]
            else:
                cells += ["%.6f" % v["mean"], "%.6f" % v["std"], str(v["n"])]
        lines.append(m + "," + ",".join(cells))
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def write_table_md(table, methods, coverage, path, scope):
    head = ("| 方法 | DISTS ↓ | MUSIQ ↑ | DOVER 技术 ↑ | DOVER 美学 ↑ |")
    sep = ("|---|---|---|---|---|")
    lines = ["# %s 指标汇总" % scope, "",
             "主表数值为 **30 个片段均值的均值**，括号内为片段间标准差；"
             "不是 3000 帧混合均值。缺失单元格留空，不写 0。", "",
             "覆盖率：%d 片段 × %d 方法 × 4 列 = %d 格，已填 %d 格。"
             % (coverage["clips"], coverage["methods"],
                coverage["expected_cells"], coverage["filled_cells"]), "",
             head, sep]
    for m in methods:
        cells = []
        for k in METRIC_COLUMNS:
            v = table[m][k]
            cells.append("—" if v is None
                         else "%.4f (%.4f)" % (v["mean"], v["std"]))
        lines.append("| %s | %s |" % (m, " | ".join(cells)))
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def write_per_clip_csv(per_clip, path):
    head = "clip,method," + ",".join(METRIC_COLUMNS)
    lines = [head]
    for r in per_clip:
        lines.append(",".join(
            [r["clip"], r["method"]] +
            ["" if r.get(k) is None else "%.6f" % r[k] for k in METRIC_COLUMNS]))
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def write_deltas(table, methods, path_csv, path_md):
    """Paired differences vs bicubic and vs edvr, on clip-level means."""
    rows = []
    for base in ("bicubic", "edvr"):
        if table.get(base, {}).get("dists") is None:
            continue
        for m in methods:
            if m == base:
                continue
            row = {"baseline": base, "method": m}
            for k in METRIC_COLUMNS:
                a, b = table[m].get(k), table[base].get(k)
                row[k] = None if (a is None or b is None) else a["mean"] - b["mean"]
            rows.append(row)

    head = "baseline,method," + ",".join("d_" + k for k in METRIC_COLUMNS)
    lines = [head]
    for r in rows:
        lines.append(",".join([r["baseline"], r["method"]] + [
            "" if r[k] is None else "%.6f" % r[k] for k in METRIC_COLUMNS]))
    open(path_csv, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    md = ["# 配对差值（相对基线，clip 级均值之差）", ""]
    for base in ("bicubic", "edvr"):
        sel = [r for r in rows if r["baseline"] == base]
        if not sel:
            continue
        md += ["## 相对 %s" % base, "",
               "| 方法 | ΔDISTS ↓好 | ΔMUSIQ ↑好 | ΔDOVER技术 ↑好 | ΔDOVER美学 ↑好 |",
               "|---|---|---|---|---|"]
        for r in sel:
            cells = ["—" if r[k] is None else "%+.4f" % r[k] for k in METRIC_COLUMNS]
            md.append("| %s | %s |" % (r["method"], " | ".join(cells)))
        md.append("")
    open(path_md, "w", encoding="utf-8").write("\n".join(md) + "\n")
    return rows


def spearman(a, b):
    """Spearman rho without scipy, so this stays dependency-light."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return None
    ra = _rank(a)
    rb = _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else None


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    # average ties
    vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            r[inv == i] = r[inv == i].mean()
    return r


def write_agreement(per_clip, path_csv, path_md):
    """Per-metric Spearman across methods, computed clip by clip."""
    by_clip = {}
    for r in per_clip:
        by_clip.setdefault(r["clip"], []).append(r)
    pairs = [("dists", "musiq"), ("dists", "dover_technical"),
             ("musiq", "dover_technical"), ("dists", "dover_aesthetic"),
             ("musiq", "dover_aesthetic")]
    rows = []
    for c, rs in sorted(by_clip.items()):
        for a, b in pairs:
            va = [r[a] for r in rs if r.get(a) is not None and r.get(b) is not None]
            vb = [r[b] for r in rs if r.get(a) is not None and r.get(b) is not None]
            rho = spearman(va, vb)
            rows.append({"clip": c, "metric_a": a, "metric_b": b,
                         "n": len(va), "spearman": rho})

    lines = ["clip,metric_a,metric_b,n,spearman"]
    for r in rows:
        lines.append("%s,%s,%s,%d,%s" % (r["clip"], r["metric_a"], r["metric_b"],
                                         r["n"], "" if r["spearman"] is None
                                         else "%.4f" % r["spearman"]))
    open(path_csv, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    md = ["# 指标间排序一致性（逐片段 Spearman，跨方法）", "",
          "每个片段内，对候选方法按两列排序求 Spearman。样本少（方法数）时 "
          "rho 波动大，只作为方向参考，不能当显著性。", "",
          "| 片段 | 指标对 | n | Spearman |", "|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s vs %s | %d | %s |"
                  % (r["clip"], r["metric_a"], r["metric_b"], r["n"],
                     "—" if r["spearman"] is None else "%+.3f" % r["spearman"]))
    md.append("")
    open(path_md, "w", encoding="utf-8").write("\n".join(md) + "\n")
    return rows


def write_rank_changes(table, methods, path_md):
    """Does adding a metric reorder the methods?  Reported per metric pair."""
    md = ["# 排序变化", "",
          "按各指标自身的“越好越靠前”方向排序。DISTS 越低越好，MUSIQ / DOVER 越高越好。",
          ""]
    rankings = {}
    for k in METRIC_COLUMNS:
        vals = [(m, table[m][k]["mean"]) for m in methods if table[m].get(k)]
        if not vals:
            continue
        vals.sort(key=lambda t: t[1], reverse=(DIRECTION[k] > 0))
        rankings[k] = [m for m, _ in vals]

    md += ["| 名次 | " + " | ".join(rankings.keys()) + " |",
           "|---" * (len(rankings) + 1) + "|"]
    n = max((len(v) for v in rankings.values()), default=0)
    for i in range(n):
        cells = [(rankings[k][i] if i < len(rankings[k]) else "—")
                 for k in rankings]
        md.append("| %d | %s |" % (i + 1, " | ".join(cells)))
    md.append("")

    md += ["## 相对 PSNR / LPIPS 的名次", ""]
    old = load_old_table()
    if old:
        md += ["| 指标 | Spearman（与 PSNR） | Spearman（与 LPIPS） |",
               "|---|---|---|"]
        for k in METRIC_COLUMNS:
            if k not in rankings:
                continue
            common_methods = [m for m in methods
                              if m in old and m in rankings[k]]
            p = spearman([old[m]["psnr"] for m in common_methods],
                         [table[m][k]["mean"] for m in common_methods])
            l = spearman([old[m]["lpips"] for m in common_methods],
                         [table[m][k]["mean"] for m in common_methods])
            md.append("| %s | %s | %s |"
                      % (k, "—" if p is None else "%+.3f" % p,
                         "—" if l is None else "%+.3f" % l))
        md.append("")
    open(path_md, "w", encoding="utf-8").write("\n".join(md) + "\n")
    return rankings


def load_old_table():
    """The stored P0 table.json: method -> {psnr, ssim, lpips, tlp, fid, ...}."""
    path = "/data/disk1/home/ghe270/p0/out/full/summary/table.json"
    if not os.path.exists(path):
        return {}
    d = json.load(open(path, encoding="utf-8"))
    if "per_method" in d:                 # p0/out/full/summary/table.json layout
        d = d["per_method"]
    elif "methods" in d:
        d = d["methods"]
    out = {}
    for m, v in d.items():
        if isinstance(v, dict) and "psnr" in v:
            out[m] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True)
    args = ap.parse_args()

    out = os.path.join(C.ROOT, "summary", args.scope)
    os.makedirs(out, exist_ok=True)

    per_clip, table, coverage, clips, methods = clip_level(args.scope)
    if not methods:
        raise SystemExit("no results for scope %s" % args.scope)

    write_table_csv(table, methods, os.path.join(out, "metrics_table.csv"))
    write_table_md(table, methods, coverage, os.path.join(out, "metrics_table.md"),
                   args.scope)
    write_per_clip_csv(per_clip, os.path.join(out, "per_clip.csv"))
    write_deltas(table, methods, os.path.join(out, "deltas.csv"),
                 os.path.join(out, "deltas.md"))
    write_agreement(per_clip, os.path.join(out, "agreement.csv"),
                    os.path.join(out, "agreement.md"))
    rankings = write_rank_changes(table, methods, os.path.join(out, "rank_changes.md"))

    C.write_json(os.path.join(out, "summary.json"),
                 {"scope": args.scope, "coverage": coverage,
                  "table": table, "rankings": rankings,
                  "n_clips": len(clips), "methods": methods})
    print("wrote %s/{metrics_table.csv,.md,per_clip.csv,deltas.csv,.md,"
          "agreement.csv,.md,rank_changes.md,summary.json}" % out)
    print("coverage: %d/%d cells, %d clips x %d methods"
          % (coverage["filled_cells"], coverage["expected_cells"],
             coverage["clips"], coverage["methods"]))


if __name__ == "__main__":
    main()
