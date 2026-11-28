# -*- coding: utf-8 -*-
"""Run DISTS / MUSIQ / DOVER over clips x methods, resumable per (clip, method).

Why a single script for B and C
-------------------------------
Stage B (3 clips x 5 methods) and stage C (30 clips x 12 methods) run the exact
same code on different scopes.  Keeping one implementation means the small run
cannot accidentally use a different protocol than the full one.

Resumability
------------
Every finished (clip, method, metric) is written into
    results/<scope>/<metric>/<clip>.json
with an atomic replace, so a killed run costs at most one clip.  A re-run skips
whatever is already complete and reports coverage instead of silently averaging
over a partial set.

Ordering
--------
Metrics are run in the roadmap's order -- DISTS, then MUSIQ, then DOVER -- so a
preprocessing mistake in one cannot contaminate the other two, and each metric
gets summarised before the next starts.

Usage
-----
    # stage B: 3 diagnostics clips, 5 methods
    python run_metrics.py --scope p3_subset --clips 000 002 009 \
        --methods bicubic bicubic_unsharp edsr edvr tecogan --metrics dists musiq dover

    # stage C: all 30 clips, all 12 methods, one metric at a time
    python run_metrics.py --scope reds30 --metrics dists
    python run_metrics.py --scope reds30 --metrics musiq
    python run_metrics.py --scope reds30 --metrics dover
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torchvision

_mod = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F          # noqa: E402

for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _mod

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

import common as C                                              # noqa: E402
import predictions as P                                         # noqa: E402

torch.backends.cudnn.benchmark = False

_GT_MUSIQ_CACHE = {}


def log(msg, fp=None):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if fp:
        fp.write(line + "\n")
        fp.flush()


def result_path(scope, metric, clip):
    return os.path.join(C.ROOT, "results", scope, metric, "%s.json" % clip)


def already_done(scope, metric, clip, method):
    path = result_path(scope, metric, clip)
    if not os.path.exists(path):
        return False
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return False
    return method in d and d[method].get("status") == "ok"


def save(scope, metric, clip, method, payload):
    path = result_path(scope, metric, clip)
    d = {}
    if os.path.exists(path):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            d = {}
    d[method] = payload
    C.write_json(path, d)


def build_metric(name, device, musiq_variant="koniq10k"):
    if name == "dists":
        return C.DistsMetric(device=device)
    if name == "musiq":
        return C.MusiqMetric(device=device, variant=musiq_variant)
    if name == "dover":
        return C.DoverMetric(device=device)
    raise ValueError(name)


def run(scope, clips, methods, metric_names, device, musiq_variant, logfp,
        force=False):
    metrics = {n: build_metric(n, device, musiq_variant) for n in metric_names}
    pairings = {n: m.pairing() for n, m in metrics.items()}

    todo_clips = []
    for clip in clips:
        need = [m for m in methods
                if force or any(not already_done(scope, n, clip, m) for n in metric_names)]
        if need:
            todo_clips.append((clip, need))
    if not todo_clips:
        log("%s: everything already complete (%d clips)" % ("+".join(metric_names),
                                                            len(clips)), logfp)
        return

    log("%s: %d/%d clips to do" % ("+".join(metric_names), len(todo_clips),
                                   len(clips)), logfp)

    for ci, (clip, need) in enumerate(todo_clips):
        t_clip = time.time()
        log("[%d/%d] clip %s -> %s  (%s)"
            % (ci + 1, len(todo_clips), clip, need, "+".join(metric_names)), logfp)
        cf = P.ClipFrames(clip, device, log=lambda s: log(s, logfp))
        gt = cf.gt

        # One method's frames are materialised once, then EVERY requested metric
        # scores them before they are dropped.  That is the only way flow-averaging
        # methods stay affordable, and it cannot change what a metric sees: each
        # metric still receives the same uint8 frames it would get in its own run.
        gt_musiq = None
        if "musiq" in metrics and clip not in _GT_MUSIQ_CACHE:
            _GT_MUSIQ_CACHE[clip] = metrics["musiq"].per_frame(gt).tolist()
        gt_musiq = _GT_MUSIQ_CACHE.get(clip)

        for m in need:
            t0 = time.time()
            try:
                pred, src = cf.frames(m)
            except FileNotFoundError as exc:
                for n in metric_names:
                    save(scope, n, clip, m,
                         {"status": "missing_source", "error": str(exc),
                          "clip": clip, "method": m})
                log("    %-20s SKIP (no prediction source): %s" % (m, exc), logfp)
                continue
            except Exception as exc:
                import traceback
                for n in metric_names:
                    save(scope, n, clip, m,
                         {"status": "error", "error": repr(exc),
                          "traceback": traceback.format_exc(),
                          "clip": clip, "method": m})
                log("    %-20s PRED ERROR: %r" % (m, exc), logfp)
                continue

            line = []
            for n in metric_names:
                if already_done(scope, n, clip, m) and not force:
                    line.append("%s=done" % n)
                    continue
                metric = metrics[n]
                try:
                    if n == "dover":
                        out = _dover_from_frames(metric, pred, clip, m)
                        payload = {"status": "ok", "metric": "dover", **out,
                                   "score": out["technical"],
                                   "pairing": pairings["dover"], "clip": clip,
                                   "method": m, "frame_ids": cf.ids,
                                   "pred_source": src,
                                   "runtime_s": round(time.time() - t0, 1)}
                    elif n == "dists":
                        vals = metric.per_frame(pred, gt)
                        payload = {
                            "status": "ok", "metric": "dists",
                            "score": float(np.mean(vals)),
                            "per_frame": [float(x) for x in vals],
                            "stats": C.summarize(vals),
                            "gt_per_frame": None, "gt_stats": None,
                            "pairing": pairings["dists"], "clip": clip, "method": m,
                            "frame_ids": cf.ids, "pred_source": src,
                            "coverage": {"n_frames": len(vals),
                                         "n_expected": len(cf.ids)},
                            "runtime_s": round(time.time() - t0, 1),
                        }
                    else:
                        vals = metric.per_frame(pred)
                        payload = {
                            "status": "ok", "metric": "musiq",
                            "score": float(np.mean(vals)),
                            "per_frame": [float(x) for x in vals],
                            "stats": C.summarize(vals),
                            "gt_per_frame": gt_musiq,
                            "gt_stats": C.summarize(gt_musiq) if gt_musiq else None,
                            "pairing": pairings["musiq"], "clip": clip, "method": m,
                            "frame_ids": cf.ids, "pred_source": src,
                            "coverage": {"n_frames": len(vals),
                                         "n_expected": len(cf.ids)},
                            "runtime_s": round(time.time() - t0, 1),
                        }
                    save(scope, n, clip, m, payload)
                    line.append("%s=%.4f" % (n, payload["score"]))
                except Exception as exc:
                    import traceback
                    save(scope, n, clip, m,
                         {"status": "error", "error": repr(exc),
                          "traceback": traceback.format_exc(),
                          "clip": clip, "method": m})
                    line.append("%s=ERR" % n)
            log("    %-20s %s  (%.0fs)  [%s]"
                % (m, "  ".join(line), time.time() - t0,
                   "cache" if src.startswith("cache") else "computed"), logfp)
            cf._product.clear()
        cf.release()
        log("  clip %s done in %.1f min" % (clip, (time.time() - t_clip) / 60), logfp)


def _dover_from_frames(dover, frames, clip, method):
    """Stage a method's frames as PNGs and score with the official DOVER path.

    A temp directory is used instead of decoding the deliverable video: H.264
    would change exactly the texture/temporal statistics DOVER is measuring.
    """
    import shutil
    import tempfile

    import cv2

    tmp = tempfile.mkdtemp(prefix="dover_%s_%s_" % (clip, method), dir="/tmp")
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, "%08d.png" % i),
                        cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        return dover.score_clip(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--metrics", nargs="*", default=["dists", "musiq", "dover"])
    ap.add_argument("--musiq-variant", default="koniq10k")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    clips = args.clips
    if clips is None:
        clips = sorted(d for d in os.listdir(C.GT_ROOT)
                       if os.path.isdir(os.path.join(C.GT_ROOT, d)))
    methods = args.methods or P.METHODS
    bad = [m for m in methods if m not in P.METHODS]
    if bad:
        raise SystemExit("unknown methods: %s" % bad)

    os.makedirs(os.path.join(C.ROOT, "logs"), exist_ok=True)
    logfp = open(os.path.join(C.ROOT, "logs",
                              "run_%s_%s.log" % (args.scope,
                                                 time.strftime("%Y%m%d_%H%M%S"))),
                 "a", encoding="utf-8")
    log("scope=%s clips=%d methods=%d metrics=%s device=%s"
        % (args.scope, len(clips), len(methods), args.metrics, args.device), logfp)

    cfg = {
        "scope": args.scope, "clips": clips, "methods": methods,
        "metrics": args.metrics, "musiq_variant": args.musiq_variant,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": C.environment_json(),
    }
    C.write_json(os.path.join(C.ROOT, "results", args.scope, "run_config.json"), cfg)

    t0 = time.time()
    run(args.scope, clips, methods, args.metrics, args.device,
        args.musiq_variant, logfp, force=args.force)
    log("metrics %s finished in %.1f min"
        % ("+".join(args.metrics), (time.time() - t0) / 60), logfp)
    for metric_name in args.metrics:
        summary(args.scope, metric_name, logfp)

    log("all metrics done", logfp)
    logfp.close()


def summary(scope, metric_name, logfp):
    """Quick coverage check after each metric -- catches a partial run early."""
    root = os.path.join(C.ROOT, "results", scope, metric_name)
    if not os.path.isdir(root):
        log("  summary %s: no results" % metric_name, logfp)
        return
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    ok, missing, err = 0, 0, 0
    per_method = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for m, v in d.items():
            st = v.get("status")
            if st == "ok":
                ok += 1
                per_method.setdefault(m, []).append(v["score"])
            elif st == "missing_source":
                missing += 1
            else:
                err += 1
    log("  summary %s: %d clips, ok=%d missing=%d error=%d"
        % (metric_name, len(files), ok, missing, err), logfp)
    for m in sorted(per_method):
        s = per_method[m]
        log("    %-20s n=%3d  mean=%.4f" % (m, len(s), float(np.mean(s))), logfp)


if __name__ == "__main__":
    main()
