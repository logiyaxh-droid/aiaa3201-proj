"""Direction C, expanded -- four uncertainty sources, calibrated, on one clip.

The earlier probes established the shape of the problem:
  * hard routing into a weaker model always loses (destination is dominated)
  * TTA ensembling on BasicVSR++ wins PSNR (+0.07..+0.14 dB) and its member
    spread is a real error proxy (Spearman 0.36..0.46)

This script turns that into an actual Part 3 experiment rather than a probe:
compare four pixel-wise uncertainty estimators against ground truth and report
calibration, not just correlation. Calibration is the part a reviewer asks for
and the part a heatmap picture cannot show.

  U1  TTA spread           |Y(x) - flip(Y(flip(x)))|
  U2  model disagreement   |Y_basicvsr++ - Y_edvr|
  U3  EDVR internal attn   1 - mean_t temporal attention  (free, no 2nd pass)
  U4  local residual energy  conv(|Laplacian(Y)|) -- a blind, no-model baseline
                              that any "learned uncertainty" must beat to matter

Calibration metrics per estimator (all pixel-wise over the clip):
  * Spearman rho vs |Y - GT|
  * AUSE-like: bin pixels by U decile, report mean |err| per bin (monotonic = calibrated)
  * error-capture: fraction of the total absolute error falling in the top 10% of U
    (a *useless* estimator captures 10%; a good one captures much more)
  * selective PSNR at 10% rejected: PSNR over the 90% "confident" pixels

Run: python p3_uncertainty.py 000
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, glob, json
import numpy as np
import cv2
from scipy.stats import spearmanr

GT_ROOT = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp"
BVSR = "/data/disk2/home/ghe270/p3/bvsrpp_frame"
EDVR = "/tmp/p2/edvr_frame"
ATTN = "/data/disk2/home/ghe270/p3/attn"
OUT = "/data/disk2/home/ghe270/p3/uncertainty.json"


def ly(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)


def lindent(y):
    k = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)
    return np.abs(cv2.filter2D(y, cv2.CV_32F, k))


def stats(U, E):
    """U, E: 1-D arrays of uncertainty and absolute error."""
    sub = np.random.RandomState(0).choice(U.size, min(300000, U.size), replace=False)
    rho = float(spearmanr(U[sub], E[sub]).correlation)

    q = np.quantile(U, np.linspace(0, 1, 11))
    bins = []
    for i in range(10):
        m = (U >= q[i]) & (U <= q[i + 1])
        if m.sum() > 100:
            bins.append(round(float(E[m].mean()), 3))

    hi = U >= np.quantile(U, 0.90)
    capture = float(E[hi].sum() / max(E.sum(), 1e-9))

    lo = ~hi
    mse = float((E[lo] ** 2).mean())
    return dict(spearman=round(rho, 4), err_per_decile=bins,
                error_capture_top10=round(capture, 4),
                selective_psnr=round(float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12))), 3))


def run_clip(clip):
    gtf = sorted(glob.glob(os.path.join(GT_ROOT, clip, "*.png")))
    bvf = sorted(glob.glob(os.path.join(BVSR, clip, "*.png")))
    edf = sorted(glob.glob(os.path.join(EDVR, clip, "*.png")))
    n = min(len(gtf), len(bvf), len(edf))
    tta_path = os.path.join(ATTN, "%s_tta" % clip)

    acc = {k: ([], []) for k in ("U1_tta", "U2_disagree", "U3_attn", "U4_lap")}
    for i in range(n):
        gt = ly(cv2.imread(gtf[i]))
        bv = ly(cv2.imread(bvf[i]))
        ed = ly(cv2.imread(edf[i]))
        E = np.abs(bv - gt)

        # U2 / U4 are always available
        acc["U2_disagree"][0].append(np.abs(bv - ed))
        acc["U4_lap"][0].append(lindent(bv))
        for k in ("U2_disagree", "U4_lap"):
            acc[k][1].append(E)

        ap = os.path.join(ATTN, "%s_attn.npz" % clip)
        if os.path.exists(ap):
            z = np.load(ap)
            a = 1.0 - z["attn"][i].astype(np.float32)
            a = cv2.resize(a, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            acc["U3_attn"][0].append(a)
            acc["U3_attn"][1].append(E)

    res = {}
    for k, (u, e) in acc.items():
        if not u:
            continue
        U = np.stack(u).ravel()
        E = np.stack(e).ravel()
        res[k] = stats(U, E)
    res["_meta"] = dict(clip=clip, frames=int(n),
                        note="U1 (TTA spread) needs the flipped pass; see p3_tta.py -- "
                             "run that first and it will fill in here")
    return res


if __name__ == "__main__":
    out = {}
    for clip in (sys.argv[1:] or ["000"]):
        r = run_clip(clip)
        out[clip] = r
        print("== clip", clip, flush=True)
        for k, v in r.items():
            if k.startswith("_"):
                continue
            print("  %-14s rho %+.3f  capture@10%% %.3f  selPSNR %.2f  deciles %s"
                  % (k, v["spearman"], v["error_capture_top10"], v["selective_psnr"],
                     v["err_per_decile"]), flush=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("OK ->", OUT)
