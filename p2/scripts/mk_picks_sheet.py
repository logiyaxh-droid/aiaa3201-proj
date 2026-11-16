"""One reference sheet for the chosen demo clips (2x2 panels per clip)."""
import os
import glob
import subprocess
import numpy as np
import cv2

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT = B + "/val_sharp"
LR = B + "/val_sharp_bicubic"
TE = "/tmp/p2/tecogan"
ED = "/tmp/p2/edvr_frame"
OUT = "/tmp/p2/demo"

CLIPS = ["015", "018", "011", "016", "013", "002"]
FRAME = int(os.environ.get("FRAME", "45"))
TW, TH = 480, 270          # half-size panels -> 960x540 per clip tile


def load(p):
    a = cv2.imread(p)
    return None if a is None else cv2.resize(a, (TW, TH), interpolation=cv2.INTER_AREA)


def put(img, t, org=(8, 24), s=0.72, c=(60, 255, 60)):
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, c, 2, cv2.LINE_AA)
    return img


tiles = []
for c in CLIPS:
    gtf = sorted(glob.glob(os.path.join(GT, c, "*.png")))
    tef = sorted(glob.glob(os.path.join(TE, c, "*.png")))
    edf = sorted(glob.glob(os.path.join(ED, c, "*.png")))
    if not (gtf and tef and edf):
        print("skip", c, len(gtf), len(tef), len(edf))
        continue
    i = min(FRAME, len(gtf) - 1, len(tef) - 1, len(edf) - 1)
    gt = load(gtf[i])
    lr = cv2.imread("%s/%s/%s" % (LR, c, os.path.basename(gtf[i])))
    bic = cv2.resize(lr, (TW, TH), interpolation=cv2.INTER_CUBIC)
    tl = put(gt.copy(), "GT")
    tr = put(bic.copy(), "Bicubic", c=(90, 210, 255))
    bl = put(load(edf[i]).copy(), "EDVR", c=(120, 255, 200))
    br = put(load(tef[i]).copy(), "TecoGAN", c=(200, 190, 255))
    tile = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])
    put(tile, "clip %s  f%d" % (c, i), (TW + 10, TH - 14), 0.8, (255, 255, 255))
    cv2.line(tile, (TW, 0), (TW, 2 * TH), (0, 0, 0), 2)
    cv2.line(tile, (0, TH), (2 * TW, TH), (0, 0, 0), 2)
    tiles.append(tile)

rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
sheet = np.vstack(rows)
p = os.path.join(OUT, "picks_sheet.png")
cv2.imwrite(p, sheet)
print("wrote", p, sheet.shape)
