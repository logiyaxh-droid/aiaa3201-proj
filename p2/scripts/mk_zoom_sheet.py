"""Zoom-sheet: for each candidate clip, show GT|Bicubic|EDVR|TecoGAN as 3x zoom crops
so we can judge by eye which clips show the biggest visible difference."""
import os
import glob
import numpy as np
import cv2

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT = B + "/val_sharp"
LR = B + "/val_sharp_bicubic"
TE = "/tmp/p2/tecogan"
ED = "/tmp/p2/edvr_frame"

CAND = os.environ.get("CAND", "003 029 028 001 002 004 005 013").split()
FRAME = int(os.environ.get("FRAME", "40"))
CW, CH = 360, 203          # crop size in HR pixels
SCALE = 2                  # 2x -> 720x406 panel
PH = CH * SCALE
PW = CW * SCALE


def lat(gt):
    g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    s = cv2.integral(e)
    H, W = e.shape
    best, bxy = -1, (0, 0)
    for y in range(0, H - CH, 32):
        for x in range(0, W - CW, 32):
            tot = s[y + CH, x + CW] - s[y, x + CW] - s[y + CH, x] + s[y, x]
            if tot > best:
                best, bxy = tot, (x, y)
    return bxy


def crop(img, xy):
    x, y = xy
    c = img[y:y + CH, x:x + CW]
    return cv2.resize(c, (PW, PH), interpolation=cv2.INTER_NEAREST)


def label(img, t):
    cv2.putText(img, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 255, 60), 2, cv2.LINE_AA)
    return img


rows = []
for c in CAND:
    gtf = sorted(glob.glob(os.path.join(GT, c, "*.png")))
    tef = sorted(glob.glob(os.path.join(TE, c, "*.png")))
    edf = sorted(glob.glob(os.path.join(ED, c, "*.png")))
    if not (gtf and tef and edf):
        print("skip", c, len(gtf), len(tef), len(edf))
        continue
    i = min(FRAME, len(gtf) - 1, len(tef) - 1, len(edf) - 1)
    gt = cv2.imread(gtf[i])
    lr = cv2.imread("%s/%s/%s" % (LR, c, os.path.basename(gtf[i])))
    bic = cv2.resize(lr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
    xy = lat(gt)
    tiles = [label(crop(gt, xy), "GT"),
             label(crop(bic, xy), "Bicubic"),
             label(crop(cv2.imread(edf[i]), xy), "EDVR"),
             label(crop(cv2.imread(tef[i]), xy), "TecoGAN")]
    sheet = np.hstack(tiles[:2] + [np.zeros((PH, 8, 3), np.uint8)] + tiles[2:])
    cv2.putText(sheet, "clip %s  f%d" % (c, i), (int(sheet.shape[1] / 2) - 90, PH - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(sheet, "clip %s  f%d" % (c, i), (int(sheet.shape[1] / 2) - 90, PH - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    rows.append(sheet)

if rows:
    cv2.imwrite("/tmp/p2/zoom_sheet.png", np.vstack(rows))
    print("ok", len(rows), np.vstack(rows).shape)
