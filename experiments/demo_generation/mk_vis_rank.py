"""Rank the 30 REDS val clips by how *visible* the SR difference is.

The metrics say EDVR is better everywhere, but "obvious demo" needs something
else: a lot of fine detail in GT that bicubic destroys, and a visible payoff
from the method. This measures exactly that.

  detail_gt   : mean |Laplacian| of the GT frame          (how much texture exists)
  loss_bic    : 1 - grad(bicubic)/grad(GT)                (how much bicubic kills)
  q_edvr      : 1 - grad(EDVR)/grad(GT)  ... reported as recovery = grad(EDVR)/grad(GT)
  recovery_edvr, recovery_teco : how much texture each method puts back
  vis_score   : sqrt(detail_gt) * loss_bic * (recovery_edvr - 1)  -- a blunt index
"""
import os
import glob
import json
import numpy as np
import cv2

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT = B + "/val_sharp"
LR = B + "/val_sharp_bicubic"
TE = "/tmp/p2/tecogan"
ED = "/tmp/p2/edvr_frame"

FRAMES = [10, 30, 50, 70, 90]


def grad_mag(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def lap_var(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)).mean())


def main():
    rows = []
    for clip in sorted(os.listdir(GT)):
        gtf = sorted(glob.glob(os.path.join(GT, clip, "*.png")))
        if len(gtf) < 95:
            continue
        dv, lv_, rb, re_, rt = [], [], [], [], []
        for i in FRAMES:
            gtp = gtf[i]
            gt = cv2.imread(gtp)
            lrp = os.path.join(LR, clip, os.path.basename(gtp))
            lr = cv2.imread(lrp)
            if lr is None:
                continue
            H, W = gt.shape[:2]
            bic = cv2.resize(lr, (W, H), interpolation=cv2.INTER_CUBIC)
            g_gt = grad_mag(gt)
            g_bi = grad_mag(bic)
            dv.append(lap_var(gt))
            lv_.append(1.0 - g_bi / g_gt)
            rb.append(g_bi / g_gt)
            edp = os.path.join(ED, clip, os.path.basename(gtp))
            tep = os.path.join(TE, clip, os.path.basename(gtp))
            if os.path.exists(edp):
                re_.append(grad_mag(cv2.imread(edp)) / g_gt)
            if os.path.exists(tep):
                rt.append(grad_mag(cv2.imread(tep)) / g_gt)
        if not dv:
            continue
        rows.append(dict(clip=clip,
                         detail=float(np.mean(dv)),
                         loss_bic=float(np.mean(lv_)),
                         rec_bic=float(np.mean(rb)),
                         rec_edvr=float(np.mean(re_)) if re_ else None,
                         rec_teco=float(np.mean(rt)) if rt else None,
                         n_edvr=len(re_)))
    rows.sort(key=lambda r: -(r["detail"] ** 0.5 * r["loss_bic"]))
    print("%-5s %8s %8s %8s %8s %6s" % ("clip", "detail", "loss_bic", "rec_edvr", "rec_teco", "nEd"))
    for r in rows:
        print("%-5s %8.1f %8.4f %8s %8s %6d" % (
            r["clip"], r["detail"], r["loss_bic"],
            ("%.4f" % r["rec_edvr"]) if r["rec_edvr"] else "-",
            ("%.4f" % r["rec_teco"]) if r["rec_teco"] else "-", r["n_edvr"]))
    json.dump(rows, open("/tmp/p2/demo/visibility_rank.json", "w"), indent=1)


if __name__ == "__main__":
    main()
