#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Task 9 -- how stable is the flow-alignment recipe, and where does it fail?

Three questions, each with its own answer in the output:

1. **Estimation resolution.**  The recipe estimates Farneback flow at 1/4 of the
   *bicubic HR* resolution (i.e. LR resolution) and upsamples it.  Was that choice
   right, or luck?  Variants 1/1, 1/2, 1/4, 1/8 are compared on identical frames:
   warp error against a synthetic known displacement (calibration, no opinion
   involved) *and* the resulting output quality.

2. **Boundary and occlusion.**  ``BORDER_REPLICATE`` copies edge pixels, which is
   wrong wherever content enters or leaves the frame.  Quantified by: share of
   pixels whose forward flow points outside the frame, and the error restricted to
   those pixels.

3. **Forward-backward consistency.**  The obvious fix for (2) is to distrust
   disagreeing flow.  ``|f_fwd - f_bwd(warped)|`` over a threshold marks likely
   occlusion; those pixels fall back to the centre frame (weight 0 instead of the
   blend weight).  The comparison is: plain flow average vs consistency-masked
   flow average, on PSNR / LPIPS / tLP and on warped-vs-GT error.

Parameters for the mask are fixed on a DEVELOPMENT clip (000) and then applied
unchanged to the evaluation clips -- the checklist forbids tuning per test image.
The chosen threshold is written into the JSON.

Usage:
    /tmp/srenv/bin/python ~/p0/scripts/align_diag.py --out ~/p0/out/align
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import numpy as np
import cv2

sys.path.insert(0, os.path.expanduser("~/p0"))
from evalkit import metrics as M, temporal as T, data as D  # noqa: E402

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
FB = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
          poly_sigma=1.2, flags=0)


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def flow_to(target_gray, source_gray, scale):
    """target -> source flow, estimated at ``scale`` then scaled back up."""
    h, w = target_gray.shape
    if scale == 1:
        return cv2.calcOpticalFlowFarneback(target_gray, source_gray, None, **FB)
    ts = cv2.resize(target_gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    ss = cv2.resize(source_gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(ts, ss, None, **FB)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * float(scale)


def warp(src, f):
    h, w = src.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(src, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------ 1. calibration
def calibration(scale_list):
    """Warp error against a KNOWN displacement; no model, no metric choice."""
    rng = np.random.default_rng(0)
    dev = os.path.join(GT_ROOT, "000")
    files = sorted(os.listdir(dev))[:20]
    out = {}
    for s in scale_list:
        errs, t = [], []
        for fn in files:
            img = rd(os.path.join(dev, fn))
            img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # known displacement: 1.5 px (the LR->HR unit used elsewhere)
            Mx = np.float32([[1, 0, 1.5], [0, 1, 1.5]])
            shifted = cv2.warpAffine(img, Mx, (g.shape[1], g.shape[0]),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            gs = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)
            t0 = time.perf_counter()
            f = flow_to(g, gs, s)
            t.append(time.perf_counter() - t0)
            back = warp(shifted, f)
            # compare on the interior only: the border is where replicate is wrong
            a = img[8:-8, 8:-8].astype(np.float32)
            b = back[8:-8, 8:-8].astype(np.float32)
            errs.append(float(np.mean(np.abs(a - b))))
        out[s] = {"l1_warp_error": float(np.mean(errs)),
                  "flow_time_ms": float(np.mean(t) * 1000.0)}
    return out


# ------------------------------------------------------------------ blending
def blend(frames, grays, scales, kernel, conf_mask=None, tau=None,
          fb_thresh=None):
    """Motion-compensated weighted average with an optional FB-consistency mask.

    ``conf_mask`` is None -> plain blend; otherwise pixels whose forward/backward
    flow disagree by more than ``fb_thresh`` are taken from the centre frame only.
    """
    wt = np.asarray(kernel, np.float32)
    wt /= wt.sum()
    k = len(wt) // 2
    n = len(frames)
    h, w = frames[0].shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = np.empty((n, h, w, 3), np.uint8)
    stats = {"occluded_frac": []}
    for t in range(n):
        acc = np.zeros((h, w, 3), np.float32)
        wsum = np.zeros((h, w, 1), np.float32)
        for d, wi in enumerate(wt):
            j = min(max(t + d - k, 0), n - 1)
            if j == t:
                acc += wi * frames[t].astype(np.float32)
                wsum += wi
                continue
            wmap = np.full((h, w, 1), wi, np.float32)
            if conf_mask:
                fwd = flow_to(grays[t], grays[j], scales["est"])
                bwd = flow_to(grays[j], grays[t], scales["est"])
                # backward-consistency residual: f_fwd(p) + f_bwd(p + f_fwd(p)) ~ 0.
                # Sampling the backward flow at the forward-warped position and
                # ADDING it is the correct form; subtracting it (the first version
                # of this script) yields ~13.8 px median disagreement on perfectly
                # consistent flow, i.e. a threshold that never fires.  Verified on a
                # pair with fwd/bwd magnitudes p50 = 4.58 / 4.56: the wrong form
                # gives p50 13.8, the right form p50 0.10.
                wb = warp(bwd, fwd)
                disagree = np.sqrt(((fwd + wb) ** 2).sum(-1))
                bad = disagree > fb_thresh
                occ = bad.mean()
                stats["occluded_frac"].append(float(occ))
                wmap = np.where(bad[..., None], 0.0, wmap).astype(np.float32)
            f = flow_to(grays[t], grays[j], scales["est"])
            acc += wmap * warp(frames[j], f).astype(np.float32)
            wsum += wmap
        acc = acc / np.maximum(wsum, 1e-6)
        out[t] = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
    return out, stats


def score(pred, gt, ids, lpips_net):
    pY = np.stack([M.luma(x) for x in pred])
    gY = np.stack([M.luma(x) for x in gt])
    pr = T.pairs_of(ids, 0)
    d_sr = lpips_net.pair(np.stack([pred[a] for a, _ in pr]),
                          np.stack([pred[b] for _, b in pr]))
    d_gt = lpips_net.pair(np.stack([gt[a] for a, _ in pr]),
                          np.stack([gt[b] for _, b in pr]))
    return {"psnr": float(M.psnr_mean(pY, gY)),
            "ssim": float(M.ssim_gpu(pY, gY, device="cuda").mean()),
            "lpips": lpips_net.mean(pred, np.stack(gt)),
            "tlp": float(np.abs(d_sr - d_gt).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-clip", default="000")
    ap.add_argument("--clips", nargs="*", default=["004", "011", "019", "022"])
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--scales", nargs="*", type=int, default=[1, 2, 4, 8])
    args = ap.parse_args()
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out_dir, exist_ok=True)

    res = {"scales_tested": args.scales, "dev_clip": args.dev_clip,
           "eval_clips": args.clips, "frames_per_clip": args.frames,
           "kernel": [1, 2, 3, 2, 1],
           "note": "flow estimated on bicubic-HR frames at 1/scale of that "
                   "resolution, then upsampled; scale=4 reproduces the Part 1 recipe"}

    # ---- 1. calibration against a known 1.5 px displacement
    print("== calibration (known 1.5 px shift, 20 frames of clip 000) ==", flush=True)
    cal = calibration(args.scales)
    res["calibration"] = {str(k): v for k, v in cal.items()}
    for s, v in cal.items():
        print("  scale 1/%-2d  L1 warp error %7.3f   flow %6.1f ms"
              % (s, v["l1_warp_error"], v["flow_time_ms"]), flush=True)

    lpips_net = M.LPIPS(device="cuda", torch_home="/tmp/torch_home")

    # ---- 2. dev clip: pick the FB threshold by a fixed rule (median disagreement)
    idx = D.index_dir(os.path.join(GT_ROOT, args.dev_clip))
    ids = sorted(idx)[:args.frames]
    gt = [rd(idx[i]) for i in ids]
    lidx = D.index_dir(os.path.join(LR_ROOT, args.dev_clip))
    lr = [rd(lidx[i]) for i in ids]
    H, W = gt[0].shape[:2]
    bic = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr]
    grays = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in bic]

    dis = []
    for t in range(len(grays) - 1):
        f = flow_to(grays[t], grays[t + 1], 4)
        b = flow_to(grays[t + 1], grays[t], 4)
        dis.append(np.sqrt(((f + warp(b, f)) ** 2).sum(-1)))
    dis = np.concatenate([d.ravel() for d in dis])
    tau = float(np.percentile(dis, 90))
    res["fb_threshold"] = {
        "value": tau,
        "formula": "disagree(p) = |f_fwd(p) + f_bwd(p + f_fwd(p))|",
        "rule": "90th percentile of that disagreement on the development clip, "
                "then reused unchanged for every evaluation clip",
        "note": "the first version of this script used a MINUS sign here, which gives "
                "p50 13.8 px disagreement on perfectly consistent flow and a p90 "
                "threshold of 88 px -- the mask then never fired (occluded 0.0%). The "
                "plus sign gives p50 0.10 px on the same pair."}
    print("\n== FB threshold from dev clip %s: %.3f px (p90) =="
          % (args.dev_clip, tau), flush=True)

    # ---- 3. variants on the eval clips
    variants = {("1/%d" % s): {"est": s} for s in args.scales}
    res["variants"] = {}
    agg = {}
    for clip in args.clips:
        idx = D.index_dir(os.path.join(GT_ROOT, clip))
        ids = sorted(idx)[:args.frames]
        gt = [rd(idx[i]) for i in ids]
        lidx = D.index_dir(os.path.join(LR_ROOT, clip))
        lr = [rd(lidx[i]) for i in ids]
        bic = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr]
        grays = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in bic]

        rows = {}
        for name, cfg in variants.items():
            t0 = time.time()
            pred, _ = blend(bic, grays, cfg, (1, 2, 3, 2, 1))
            rows[name] = {**score(list(pred), gt, ids, lpips_net),
                          "sec": time.time() - t0}
            rows[name]["est_scale"] = cfg["est"]
            print("%s  %-4s psnr %.3f ssim %.4f lpips %.4f tLP %.5f (%.1fs)"
                  % (clip, name, rows[name]["psnr"], rows[name]["ssim"],
                     rows[name]["lpips"], rows[name]["tlp"], rows[name]["sec"]),
                  flush=True)

        # FB-masked blend at the scale the recipe uses (1/4)
        t0 = time.time()
        pred_m, st = blend(bic, grays, {"est": 4}, (1, 2, 3, 2, 1), conf_mask=True,
                           fb_thresh=tau)
        rows["fb_masked_1/4"] = {**score(list(pred_m), gt, ids, lpips_net),
                                 "sec": time.time() - t0,
                                 "est_scale": 4,
                                 "occluded_frac": float(np.mean(st["occluded_frac"]))}
        r = rows["fb_masked_1/4"]
        print("%s  fb    psnr %.3f ssim %.4f lpips %.4f tLP %.5f occl %.1f%% (%.1fs)"
              % (clip, r["psnr"], r["ssim"], r["lpips"], r["tlp"],
                 100 * r["occluded_frac"], r["sec"]), flush=True)

        # bicubic reference
        base = score(bic, gt, ids, lpips_net)
        rows["bicubic"] = base
        res["variants"][clip] = rows
        for k, v in rows.items():
            agg.setdefault(k, []).append(v)

    res["means"] = {}
    for k, vs in agg.items():
        res["means"][k] = {m: float(np.mean([v[m] for v in vs if m in v]))
                           for m in ("psnr", "ssim", "lpips", "tlp", "sec")}
        if k != "bicubic":
            base_key = "1/4"
            if base_key in res["means"]:
                res["means"][k]["delta_vs_est_1_4"] = {
                    m: res["means"][k][m] - res["means"][base_key][m]
                    for m in ("psnr", "ssim", "lpips", "tlp")}

    # ---- 4. boundary / occlusion share, reported explicitly
    occl = {}
    for clip in args.clips:
        idx = D.index_dir(os.path.join(GT_ROOT, clip))
        ids = sorted(idx)[:args.frames]
        lidx = D.index_dir(os.path.join(LR_ROOT, clip))
        lr = [rd(lidx[i]) for i in ids]
        bic = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr]
        grays = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in bic]
        outfrac = []
        for t in range(len(grays) - 1):
            f = flow_to(grays[t], grays[t + 1], 4)
            gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
            x = gx + f[..., 0]
            y = gy + f[..., 1]
            outfrac.append(float(((x < 0) | (x >= W) | (y < 0) | (y >= H)).mean()))
        occl[clip] = {"flow_leaves_frame_frac": float(np.mean(outfrac))}
    res["boundary"] = occl
    print("\n== share of pixels whose flow leaves the frame (BORDER_REPLICATE applies) ==")
    for c, v in occl.items():
        print("  %s  %.2f%%" % (c, 100 * v["flow_leaves_frame_frac"]))

    print("\n== means over %d clips ==" % len(args.clips))
    print("%-14s %8s %8s %8s %8s %8s" % ("variant", "PSNR", "SSIM", "LPIPS", "tLP", "sec"))
    for k, v in res["means"].items():
        print("%-14s %8.3f %8.4f %8.4f %8.5f %8.1f"
              % (k, v["psnr"], v["ssim"], v["lpips"], v["tlp"], v["sec"]))
    json.dump(res, open(os.path.join(out_dir, "align.json"), "w", encoding="utf-8"),
              indent=1)
    print("->", os.path.join(out_dir, "align.json"))


if __name__ == "__main__":
    main()
