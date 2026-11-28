# -*- coding: utf-8 -*-
"""Stage D driver: regional attribution, motion grouping, temporal anomalies,
and the metric-sensitivity checks -- only where stage C found something.

The point of this file is to stop the report from saying "method X has a lower
DISTS" and instead say "method X loses 0.05 DISTS on 32x32 blocks where the
high-frequency energy is low but the optical flow is large, in clips 004 and 017,
at frames 62-71".  Every row it emits carries clip id, frame id and block
coordinates so the claim can be checked by looking at one image.

What is deliberately NOT done here
----------------------------------
* Block-level DISTS is computed by cropping and is labelled 局部裁切诊断 in every
  output.  It is a locator, never the headline score.
* Motion groups come from the existing tercile split when that file exists; the
  between-group statement is limited by however many clips each group holds, and
  that count is printed.
* Nothing here trains, fine-tunes, or alters a prediction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import common as C
import predictions as P
from stageD_blocks import (block_grid, block_stats, flow_between,
                           flow_consistency, load_metric, old_results)

BLOCK = 32
CASES = os.path.join(C.ROOT, "cases")
MOTION_FILE = "/data/disk1/home/ghe270/p0/out/motion/motion_groups.json"


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------- D1

def d1_regional(scope, clips, methods, out_dir):
    """Block-level attribution for the methods that differ most in DISTS."""
    dists = load_metric(scope, "dists")
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for clip in clips:
        if clip not in dists:
            continue
        cf = P.ClipFrames(clip, log=lambda s: None)
        ids = cf.ids
        h, w = cf.gt[0].shape[:2]
        blocks = block_grid(h, w, BLOCK)
        gt = cf.gt

        # per (method, frame): block statistics + flow stats
        for m in methods:
            try:
                pred, _ = cf.frames(m)
            except Exception as exc:
                log("  clip %s %s: no prediction (%s)" % (clip, m, exc))
                continue
            for t in range(0, len(ids), 5):        # every 5th frame is plenty
                bs = block_stats(pred[t], gt[t], blocks, BLOCK)
                if t > 0:
                    fwd, _, cons = flow_consistency(gt[t - 1], gt[t])
                    mag = np.sqrt(fwd[..., 0] ** 2 + fwd[..., 1] ** 2)
                else:
                    cons = np.zeros((h, w), np.float32)
                    mag = np.zeros((h, w), np.float32)
                for b in bs:
                    y, x = b["y"], b["x"]
                    b["flow_mag"] = float(mag[y:y + BLOCK, x:x + BLOCK].mean())
                    b["fb_consistency"] = float(
                        cons[y:y + BLOCK, x:x + BLOCK].mean())
                for b in bs:
                    b.update({"clip": clip, "method": m, "frame_id": ids[t],
                              "block": BLOCK})
                    rows.append(b)
            cf._product.clear()
        cf.release()
        log("  clip %s blocks done (%d rows so far)" % (clip, len(rows)))

    path = os.path.join(out_dir, "d1_blocks.json")
    C.write_json(path, {"block_size": BLOCK, "n_rows": len(rows),
                        "note": "block metrics are cropped-input diagnostics, "
                                "not the frame-level headline score",
                        "rows": rows[:200000]})
    log("  wrote %s (%d rows)" % (path, len(rows)))
    return rows


# ------------------------------------------------------------------- D2

def d2_motion(scope, out_dir):
    """Per-clip motion terciles x method x metric, with group counts shown."""
    os.makedirs(out_dir, exist_ok=True)
    dists = load_metric(scope, "dists")
    musiq = load_metric(scope, "musiq")
    dover = load_metric(scope, "dover")

    motion = {}
    if os.path.exists(MOTION_FILE):
        raw = json.load(open(MOTION_FILE, encoding="utf-8"))
        motion = _flatten_motion(raw)

    if not motion:
        # Recompute mean flow magnitude per clip from GT; cheap enough (10 frames).
        log("  no motion_groups.json, recomputing per-clip mean flow")
        for clip in sorted(dists):
            gt_dir = os.path.join(C.GT_ROOT, clip)
            ids = C.clip_frame_ids(gt_dir)
            frames = C.read_clip(gt_dir, ids[:11])
            mags = []
            for a, b in zip(frames[:-1], frames[1:]):
                f = flow_between(a, b)
                mags.append(float(np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2).mean()))
            motion[clip] = float(np.mean(mags))

    vals = np.array([motion[c] for c in motion])
    if len(vals) >= 3:
        q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    else:
        q1 = q2 = float(np.median(vals)) if len(vals) else 0.0
    group_of = {c: ("low" if v <= q1 else "high" if v > q2 else "mid")
                for c, v in motion.items()}

    methods = sorted({m for c in dists for m in dists[c]})
    table = {}
    for m in methods:
        for g in ("low", "mid", "high"):
            sel = [c for c in dists if group_of.get(c) == g and m in dists[c]]
            if not sel:
                continue
            row = {"n_clips": len(sel)}
            row["dists"] = float(np.mean([dists[c][m]["score"] for c in sel]))
            mv = [musiq[c][m]["score"] for c in sel if c in musiq and m in musiq[c]]
            dv = [dover[c][m]["technical"] for c in sel
                  if c in dover and m in dover[c]]
            row["musiq"] = float(np.mean(mv)) if mv else None
            row["dover_technical"] = float(np.mean(dv)) if dv else None
            table.setdefault(m, {})[g] = row

    out = {"motion_source": MOTION_FILE if os.path.exists(MOTION_FILE)
           else "recomputed from GT",
           "terciles": {"q1": float(q1), "q2": float(q2)},
           "group_sizes": {g: sum(1 for c in group_of if group_of[c] == g)
                           for g in ("low", "mid", "high")},
           "per_method_group": table,
           "clip_groups": group_of,
           "caveat": "three diagnostics clips cannot support a between-group "
                     "claim; group sizes are reported so the reader can judge."}
    C.write_json(os.path.join(out_dir, "d2_motion_groups.json"), out)

    md = ["# 运动强度分组", "",
          "三分位阈值 q1=%.3f  q2=%.3f；各组片段数 %s。"
          % (q1, q2, json.dumps(out["group_sizes"], ensure_ascii=False)), "",
          "| 方法 | 组 | n | DISTS | MUSIQ | DOVER技术 |",
          "|---|---|---|---|---|---|"]
    for m in sorted(table):
        for g in ("low", "mid", "high"):
            r = table[m].get(g)
            if not r:
                continue
            md.append("| %s | %s | %d | %.4f | %s | %s |"
                      % (m, g, r["n_clips"], r["dists"],
                         "—" if r["musiq"] is None else "%.2f" % r["musiq"],
                         "—" if r["dover_technical"] is None
                         else "%.4f" % r["dover_technical"]))
    md.append("")
    open(os.path.join(out_dir, "d2_motion_groups.md"), "w",
           encoding="utf-8").write("\n".join(md) + "\n")
    log("  wrote d2_motion_groups.{json,md}; group sizes %s"
        % out["group_sizes"])
    return out


def _flatten_motion(raw):
    """Accept several plausible shapes and return {clip: mean_flow_magnitude}."""
    out = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, (int, float)):
                out[str(k)] = float(v)
            elif isinstance(v, dict):
                for key in ("flow_mag_mean", "mean_flow", "mean_magnitude",
                            "motion", "mean"):
                    if key in v and isinstance(v[key], (int, float)):
                        out[str(k)] = float(v[key])
                        break
                else:
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)) and "flow" in kk.lower():
                            out[str(k)] = float(vv)
                            break
    return out


# ------------------------------------------------------------------- D3

def d3_temporal(scope, out_dir, worst_n=6):
    """Line up DOVER technical with tLP / flow-warp LPIPS / frame differences."""
    os.makedirs(out_dir, exist_ok=True)
    dover = load_metric(scope, "dover")
    ranks = []
    for clip, rows in dover.items():
        for m, v in rows.items():
            ranks.append((v["technical"], clip, m))
    if not ranks:
        log("  no dover rows, skipping D3")
        return {}
    ranks.sort()
    worst = ranks[:worst_n]

    rows = []
    for score, clip, m in worst:
        old = old_results(clip).get(m, {})
        rows.append({
            "clip": clip, "method": m,
            "dover_technical": score,
            "dover_aesthetic": dover[clip][m]["aesthetic"],
            "tlp_p0": old.get("tlp"),
            "flow_warp_lpips_p0": old.get("flow_warp_lpips"),
            "psnr_p0": old.get("psnr"),
            "lpips_p0": old.get("lpips"),
        })

    out = {"criterion": "lowest DOVER technical",
           "n": len(rows), "rows": rows,
           "caveat": "DOVER is one value per clip and cannot say which frame "
                     "flickers; the per-pair tLP column is what locates frames."}
    C.write_json(os.path.join(out_dir, "d3_temporal.json"), out)

    md = ["# 时序异常候选（DOVER 技术分最低的 %d 个 片段×方法）" % len(rows), "",
          "DOVER 是整段分数，不能指出哪一帧闪烁；tLP 与 flow-warp LPIPS 来自 P0 的逐相邻帧结果。", "",
          "| 片段 | 方法 | DOVER技术 | DOVER美学 | tLP | flow-warp LPIPS | PSNR | LPIPS |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s | %.4f | %.4f | %s | %s | %s | %s |"
                  % (r["clip"], r["method"], r["dover_technical"],
                     r["dover_aesthetic"],
                     "—" if r["tlp_p0"] is None else "%.5f" % r["tlp_p0"],
                     "—" if r["flow_warp_lpips_p0"] is None
                     else "%.5f" % r["flow_warp_lpips_p0"],
                     "—" if r["psnr_p0"] is None else "%.3f" % r["psnr_p0"],
                     "—" if r["lpips_p0"] is None else "%.4f" % r["lpips_p0"]))
    md.append("")
    open(os.path.join(out_dir, "d3_temporal.md"), "w",
           encoding="utf-8").write("\n".join(md) + "\n")
    log("  wrote d3_temporal.{json,md} (%d rows)" % len(rows))
    return out


# ------------------------------------------------------------------- D4

def d4_sensitivity(clip, methods, out_dir, device="cuda"):
    """Second MUSIQ weight and repeated DOVER seeds on representative clips."""
    os.makedirs(out_dir, exist_ok=True)
    out = {"clip": clip, "methods": methods, "musiq": {}, "dover": {}}

    log("  D4 MUSIQ second weight (paq2piq) vs primary (koniq10k)")
    m1 = C.MusiqMetric(device=device, variant="koniq10k")
    m2 = C.MusiqMetric(device=device, variant="paq2piq")
    for m in methods:
        cf = P.ClipFrames(clip, device, log=lambda s: None)
        try:
            pred, _ = cf.frames(m)
        except Exception as exc:
            log("    %s: no prediction (%s)" % (m, exc))
            cf.release()
            continue
        a = m1.per_frame(pred)
        b = m2.per_frame(pred)
        out["musiq"][m] = {"koniq10k": float(a.mean()),
                           "paq2piq": float(b.mean()),
                           "per_frame_spearman": _spearman(a.tolist(), b.tolist())}
        log("    %-20s koniq10k %.3f  paq2piq %.3f"
            % (m, a.mean(), b.mean()))
        cf._product.clear()
        cf.release()

    log("  D4 DOVER seed repeatability")
    for m in methods:
        cf = P.ClipFrames(clip, device, log=lambda s: None)
        try:
            pred, _ = cf.frames(m)
        except Exception:
            cf.release()
            continue
        scores = []
        for seed in (C.DOVER_SEED, C.DOVER_SEED + 1, C.DOVER_SEED + 2):
            d = C.DoverMetric(device=device, seed=seed)
            import tempfile, shutil
            tmp = tempfile.mkdtemp(prefix="d4_%s_%s_" % (clip, m), dir="/tmp")
            try:
                for i, f in enumerate(pred):
                    cv2.imwrite(os.path.join(tmp, "%08d.png" % i),
                                cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                s = d.score_clip(tmp)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            scores.append({"seed": seed, "technical": s["technical"],
                           "aesthetic": s["aesthetic"]})
            del d
        out["dover"][m] = scores
        tech = [s["technical"] for s in scores]
        log("    %-20s technical %s  spread %.5f"
            % (m, ["%.4f" % t for t in tech], max(tech) - min(tech)))
        cf._product.clear()
        cf.release()

    C.write_json(os.path.join(out_dir, "d4_sensitivity_%s.json" % clip), out)
    log("  wrote d4_sensitivity_%s.json" % clip)
    return out


def _spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3:
        return None
    ra = _rank(a) - _rank(a).mean()
    rb = _rank(b) - _rank(b).mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else None


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="reds30")
    ap.add_argument("--clips", nargs="*", default=["000", "004", "017"])
    ap.add_argument("--methods", nargs="*", default=["bicubic", "edvr", "tecogan"])
    ap.add_argument("--parts", nargs="*", default=["d1", "d2", "d3"])
    ap.add_argument("--d4-clip", default=None)
    args = ap.parse_args()

    base = os.path.join(CASES, args.scope)
    os.makedirs(base, exist_ok=True)

    if "d1" in args.parts:
        log("D1 regional attribution on %s" % args.clips)
        d1_regional(args.scope, args.clips, args.methods, base)
    if "d2" in args.parts:
        log("D2 motion grouping")
        d2_motion(args.scope, base)
    if "d3" in args.parts:
        log("D3 temporal anomalies")
        d3_temporal(args.scope, base)
    if "d4" in args.parts:
        clip = args.d4_clip or args.clips[0]
        log("D4 sensitivity on clip %s" % clip)
        d4_sensitivity(clip, args.methods, base)


if __name__ == "__main__":
    main()
