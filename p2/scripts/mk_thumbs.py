"""GT thumbnails for clips that have no EDVR frames yet, to pick demos."""
import os
import glob
import sys
import numpy as np
import cv2

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT = B + "/val_sharp"
clips = sys.argv[1:] or ["011", "014", "015", "016", "017", "018", "023", "005"]
tiles = []
for c in clips:
    fs = sorted(glob.glob(os.path.join(GT, c, "*.png")))
    for fr in (20, 70):
        im = cv2.imread(fs[fr])
        im = cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(im, "%s f%d" % (c, fr), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(im, "%s f%d" % (c, fr), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        tiles.append(im)
rows = []
for i in range(0, len(tiles), 4):
    r = tiles[i:i + 4]
    while len(r) < 4:
        r.append(np.zeros_like(tiles[0]))
    rows.append(np.hstack(r))
cv2.imwrite("/tmp/p2/gt_thumbs.png", np.vstack(rows))
print("ok", np.vstack(rows).shape)
