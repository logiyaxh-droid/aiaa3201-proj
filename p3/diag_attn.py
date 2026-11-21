"""Diagnostic: is EDVR's internal attention actually correlated with EDVR's error?

If yes, Direction C has a free, calibrated-enough uncertainty source and the
whole route can skip a second forward pass. If no, we need test-time
augmentation or model disagreement instead -- better to know now than after
building the router.

Measures, per frame, on the Y channel:
  err   = |EDVR_Y - GT_Y|                  (HR grid, x4)
  dcorr = 1 - mean_t corr_prob              (LR grid, x4)
  dattn = 1 - attn                          (LR grid, x4)
and reports Spearman correlation between err and each signal, plus the mean
error in the top/bottom 10% of each signal (a routing gain, not just a
correlation).
"""
import os, sys, glob, json
import numpy as np
import cv2
from scipy.stats import spearmanr

GT = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp"
LR = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp_bicubic"
EDVR = "/tmp/p2/edvr_frame"
ATTN = "/data/disk2/home/ghe270/p3/attn"

clips = sys.argv[1:] or ["000"]
out = {}
for clip in clips:
    z = np.load(os.path.join(ATTN, "%s_attn.npz" % clip))
    corr = z["corr"].astype(np.float32)
    attn = z["attn"].astype(np.float32)
    gts = sorted(glob.glob(os.path.join(GT, clip, "*.png")))
    eds = sorted(glob.glob(os.path.join(EDVR, clip, "*.png")))
    n = min(len(corr), len(eds))
    rows = []
    for i in range(n):
        gt = cv2.imread(gts[i])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        pr = cv2.imread(eds[i])
        pr = cv2.cvtColor(pr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        H, W = gt.shape
        err = np.abs(pr - gt)
        dc = cv2.resize(1.0 - corr[i], (W, H), interpolation=cv2.INTER_NEAREST)
        da = cv2.resize(1.0 - attn[i], (W, H), interpolation=cv2.INTER_NEAREST)
        rows.append((err, dc, da))
    E = np.stack([r[0] for r in rows]).ravel()
    DC = np.stack([r[1] for r in rows]).ravel()
    DA = np.stack([r[2] for r in rows]).ravel()
    sub = np.random.RandomState(0).choice(len(E), min(600000, len(E)), replace=False)
    rc = spearmanr(E[sub], DC[sub]).correlation
    ra = spearmanr(E[sub], DA[sub]).correlation

    def gain(S):
        lo, hi = np.quantile(S, 0.10), np.quantile(S, 0.90)
        return float(E[S <= lo].mean()), float(E[S >= hi].mean())

    out[clip] = dict(frames=int(n), spearman_corr=float(rc), spearman_attn=float(ra),
                     err_low_corr=gain(DC), err_high_corr=gain(DC),
                     err_low_attn=gain(DA), err_high_attn=gain(DA),
                     err_mean=float(E.mean()))
    print(clip, json.dumps(out[clip], indent=1), flush=True)

json.dump(out, open("/data/disk2/home/ghe270/p3/attn_diag.json", "w"), indent=2)
print("OK")
