# -*- coding: utf-8 -*-
"""Score a REDS-Real c9 candidate directory, with temporal diagnostics.

Same metric stack as the rest of the study (luma PSNR / SSIM / LPIPS / DISTS /
tLP) plus the two diagnostics needed to judge a zero-shot chain:

  tLP decomposition   d_SR = LPIPS(SR[t-1], SR[t]), d_GT = LPIPS(GT[t-1], GT[t]).
                      tLP = |d_SR - d_GT| hides *which* side moved, so both
                      terms are stored.  A chain that raises PSNR while pushing
                      d_SR far below d_GT is smoothing/flickering differently
                      from the ground truth even if the absolute value is small.

  occlusion-aware E_warp
                      For each consecutive pair, warp the previous frame onto the
                      current one with Farneback flow, then score only the pixels
                      that actually see data (forward-backward consistency):
                          |f_fwd(p) + f_bwd(p + f_fwd(p))| <= tol
                      plus a border margin.  Reported for both the SR sequence and
                      the GT sequence; E_warp_gap = SR - GT says whether the
                      method is temporally *less* stable than the reference.
                      Sign note: the consistency test is a SUM, not a difference
                      -- written as a difference it reports ~14 px on identical
                      flow and the mask never fires (documented in p0).

Usage:
  python score_reds_real.py --name rbv_osediff --dir outputs/reds_real_c9/rbv_osediff
  python score_reds_real.py --name all --dir A --also B C      # compare several
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")
sys.path.insert(0, "/data/disk1/home/ghe270/p3/quality_metrics/scripts")

os.environ.setdefault(
    "TORCH_HOME", "/data/disk1/home/ghe270/p3/quality_metrics/weights/torch_home")

DATA = os.environ.get("RBVSR_DATA") or os.path.join(ROOT, "data", "reds_real_c9")
GT = os.path.join(DATA, "GT")
METRIC_KEYS = ("psnr", "ssim", "lpips", "dists", "tlp")


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def read_rgb(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------ temporal diagnostics

def _flow(target_gray, source_gray, scale=4):
    """Flow pulling `source` onto `target` (direction calibrated in p0)."""
    h, w = target_gray.shape
    if scale == 1:
        return cv2.calcOpticalFlowFarneback(target_gray, source_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
    ts = cv2.resize(target_gray, (w // scale, h // scale),
                    interpolation=cv2.INTER_AREA)
    ss = cv2.resize(source_gray, (w // scale, h // scale),
                    interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(ts, ss, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * float(scale)


def _e_warp_pair(prev, cur, tol=3.0, margin=8, scale=4):
    """Occlusion-aware warping error between two RGB frames of one sequence.

    Returns (error, valid_ratio, median_flow).  `error` is the mean absolute
    difference in 0..255 over pixels whose flow is forward-backward consistent;
    pixels near the frame border or occluded are excluded rather than clipped.
    """
    if tol is None:
        tol = 3.0
    g_prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    g_cur = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
    h, w = g_prev.shape

    f_fwd = _flow(g_cur, g_prev, scale)       # prev -> cur
    f_bwd = _flow(g_prev, g_cur, scale)       # cur -> prev
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    # forward-backward consistency: |f_fwd(p) + f_bwd(p + f_fwd(p))| ~ 0
    bx = np.clip(gx + f_fwd[..., 0], 0, w - 1).astype(np.float32)
    by = np.clip(gy + f_fwd[..., 1], 0, h - 1).astype(np.float32)
    b_back = cv2.remap(f_bwd, bx, by, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    fb = np.abs(f_fwd + b_back).sum(axis=2)

    warped = cv2.remap(prev, gx + f_fwd[..., 0], gy + f_fwd[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    valid = fb <= tol
    if margin > 0:
        valid[:margin, :] = False
        valid[-margin:, :] = False
        valid[:, :margin] = False
        valid[:, -margin:] = False
    if not valid.any():
        return None, 0.0, float(np.median(np.abs(f_fwd).sum(axis=2)))
    diff = np.abs(warped.astype(np.float32) - cur.astype(np.float32)).mean(axis=2)
    return (float(diff[valid].mean()), float(valid.mean()),
            float(np.median(np.abs(f_fwd).sum(axis=2))))


def temporal_diag(pred, gts, ids, lp, ev_override=None):
    """tLP decomposition plus occlusion-aware E_warp for one clip."""
    from evalkit import temporal as ET

    pairs = ET.pairs_of(ids, 0)
    out = {"n_pairs": len(pairs)}
    if not pairs:
        out.update({"tlp": None, "d_sr": None, "d_gt": None,
                    "e_warp_sr": None, "e_warp_gt": None})
        return out

    sr_prev = np.stack([pred[a] for a, _ in pairs])
    sr_cur = np.stack([pred[b] for _, b in pairs])
    gt_prev = np.stack([gts[a] for a, _ in pairs])
    gt_cur = np.stack([gts[b] for _, b in pairs])
    d_sr = lp.pair(sr_prev, sr_cur)
    d_gt = lp.pair(gt_prev, gt_cur)
    out.update({"tlp": float(np.abs(d_sr - d_gt).mean()),
                "d_sr": float(d_sr.mean()), "d_gt": float(d_gt.mean()),
                "d_sr_minus_d_gt": float((d_sr - d_gt).mean())})

    es, eg, ratios, flows = [], [], [], []
    for a, b in pairs:
        e_s, r_s, med = _e_warp_pair(pred[a], pred[b], ev_override)
        e_g, _, _ = _e_warp_pair(gts[a], gts[b], ev_override)
        if e_s is not None and e_g is not None:
            es.append(e_s)
            eg.append(e_g)
            ratios.append(r_s)
            flows.append(med)
    if es:
        out.update({"e_warp_sr": float(np.mean(es)),
                    "e_warp_gt": float(np.mean(eg)),
                    "e_warp_gap": float(np.mean(es) - np.mean(eg)),
                    "valid_ratio": float(np.mean(ratios)),
                    "flow_median_px": float(np.mean(flows))})
    else:
        out.update({"e_warp_sr": None, "e_warp_gt": None,
                    "e_warp_gap": None})
    return out


# ------------------------------------------------------------------------ scoring

def score_dir(name, path, lp, dists, dev):
    from evalkit import metrics as EM

    clips = sorted(d for d in os.listdir(GT)
                   if os.path.isdir(os.path.join(GT, d)))
    rows, missing = {}, []
    all_y_p, all_y_g = [], []
    for clip in clips:
        ids = sorted(int(os.path.splitext(f)[0])
                     for f in os.listdir(os.path.join(GT, clip))
                     if f.lower().endswith(".png"))
        preds, gts, keep = [], [], []
        for fid in ids:
            pp = os.path.join(path, clip, "%s.png" % _fmt(fid, path, clip))
            if not os.path.exists(pp):
                # frame ids are zero padded to 8 in this dataset
                pp = os.path.join(path, clip, "%08d.png" % fid)
            if not os.path.exists(pp):
                missing.append("%s/%d" % (clip, fid))
                continue
            p = read_rgb(pp)
            g = read_rgb(os.path.join(GT, clip, "%08d.png" % fid))
            if p.shape != g.shape:
                raise SystemExit("%s/%d size %s vs GT %s"
                                 % (clip, fid, p.shape, g.shape))
            preds.append(p)
            gts.append(g)
            keep.append(fid)
        if not preds:
            continue

        pY = np.stack([EM.luma(x) for x in preds])
        gY = np.stack([EM.luma(x) for x in gts])
        row = {
            "psnr": float(EM.psnr_mean(pY, gY)),
            "ssim": float(EM.ssim_gpu(pY, gY, device=dev).mean()),
            "lpips": float(lp.pair(np.stack(preds), np.stack(gts)).mean()),
            "dists": float(dists.per_frame(preds, gts).mean()),
            "n_frames": len(preds),
            "per_frame": {"ids": keep,
                          "psnr": [float(v) for v in
                                   (EM.psnr_mean(pY[i:i + 1], gY[i:i + 1])
                                    for i in range(len(keep)))]},
        }
        row["temporal"] = temporal_diag(preds, gts, keep, lp)
        row["tlp"] = row["temporal"]["tlp"]
        rows[clip] = row
        all_y_p.append(pY)
        all_y_g.append(gY)
        print("%s %s PSNR %.4f SSIM %.4f LPIPS %.5f tLP %s "
              "(d_SR %.4f d_GT %.4f) E_warp SR %.3f GT %.3f"
              % (name, clip, row["psnr"], row["ssim"], row["lpips"],
                 "%.5f" % row["tlp"] if row["tlp"] is not None else "N/A",
                 row["temporal"].get("d_sr") or -1,
                 row["temporal"].get("d_gt") or -1,
                 row["temporal"].get("e_warp_sr") or -1,
                 row["temporal"].get("e_warp_gt") or -1), flush=True)

    if not rows:
        raise SystemExit("no scored frames under %s" % path)
    mean = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in rows.values() if r[k] is not None]
        mean[k] = float(np.mean(vals)) if vals else None
    for k in ("d_sr", "d_gt", "d_sr_minus_d_gt", "e_warp_sr", "e_warp_gt",
              "e_warp_gap", "valid_ratio", "flow_median_px"):
        vals = [r["temporal"][k] for r in rows.values()
                if r["temporal"].get(k) is not None]
        mean[k] = float(np.mean(vals)) if vals else None
    mean["psnr_allframes"] = float(
        EM.psnr_mean(np.concatenate(all_y_p), np.concatenate(all_y_g)))
    mean["n_clips"] = len(rows)
    mean["n_frames"] = sum(r["n_frames"] for r in rows.values())

    return {"name": name, "dir": os.path.abspath(path), "missing": missing,
            "clip_means": rows, "mean": mean,
            "protocol": {
                "dataset": DATA,
                "metrics": "luma PSNR/SSIM/LPIPS/DISTS against GT; "
                           "tLP with d_SR/d_GT decomposition; "
                           "occlusion-aware E_warp on FB-consistent pixels",
                "e_warp": {"tol": 3.0, "margin": 8, "flow_scale": 4,
                           "consistency": "|f_fwd(p)+f_bwd(p+f_fwd(p))| <= tol"},
            }}


def _fmt(fid, path, clip):
    """Match the padding actually used on disk (all files are 8-digit here)."""
    return "%08d" % fid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--also", nargs="*", default=[])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from evalkit import metrics as EM
    import common as C

    dev = os.environ.get("RVQ_DEV", "cuda")
    lp = EM.LPIPS(device=dev, torch_home="/tmp/torch_home")
    dists = C.DistsMetric(device=dev)

    names = [args.name] + list(args.also)
    dirs = [args.dir] + [os.path.join(os.path.dirname(args.dir.rstrip("/")), n)
                         for n in args.also]
    # Results always land next to the dataset, not next to the prediction dir:
    # `--dir outputs/reds_real_c9/x` used to write into results/reds_real/.
    res_root = os.path.join(ROOT, "results",
                            os.path.basename(os.path.abspath(DATA)))
    out = {}
    for name, path in zip(names, dirs):
        res = score_dir(name, path, lp, dists, dev)
        out[name] = res
        dst = (args.out or os.path.join(res_root, name, "metrics.json"))
        dump(dst, res)
        m = res["mean"]
        print("\n%s MEAN PSNR %.4f SSIM %.4f LPIPS %.5f DISTS %.5f tLP %.5f "
              "d_SR %.4f d_GT %.4f E_warp %.3f (GT %.3f) gap %+.3f"
              % (name, m["psnr"], m["ssim"], m["lpips"], m["dists"],
                 m["tlp"] or -1, m["d_sr"] or -1, m["d_gt"] or -1,
                 m["e_warp_sr"] or -1, m["e_warp_gt"] or -1,
                 m["e_warp_gap"] or 0), flush=True)
        print("wrote", dst, flush=True)


if __name__ == "__main__":
    main()
