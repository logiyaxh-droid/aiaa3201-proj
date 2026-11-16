import os, glob, json
import numpy as np
import cv2

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT = B + "/val_sharp"
cands = ["000", "001", "002", "003", "004", "005", "009", "013", "026", "028", "029"]
idx = 20
tiles = []
for c in cands:
    fs = sorted(glob.glob(os.path.join(GT, c, "*.png")))
    im = cv2.imread(fs[idx])
    im = cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.putText(im, "clip %s" % c, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    tiles.append(im)
rows = []
for i in range(0, len(tiles), 3):
    row = tiles[i:i + 3]
    while len(row) < 3:
        row.append(np.zeros_like(tiles[0]))
    rows.append(np.hstack(row))
cv2.imwrite("/tmp/p2/cands_gt.png", np.vstack(rows))
print("ok")
