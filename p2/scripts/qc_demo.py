#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sub-second QC of the demo renders + per-source visibility ranking.

Checks every produced mp4 against the reference format (1920x1080, 24 fps, h264,
yuv420p, 100 frames) and writes picks_sheet.png: one 2x2 tile per source clip so the
user can see at a glance which clips actually look different.
"""
import os, json, glob, subprocess
import numpy as np
import cv2

DEMO = "/tmp/wild/demo"
FIG = "/tmp/wild/out/figures"
SRC = {"A": "outdoor telephoto", "B": "indoor low-light", "C": "dark corridor",
       "D": "night shrubs", "E": "night bikes"}
WANT = {"codec_name": "h264", "width": 1920, "height": 1080, "pix_fmt": "yuv420p",
        "r_frame_rate": "24/1", "nb_frames": 100}


def probe(p):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames",
           "-of", "json", p]
    out = subprocess.run(cmd, capture_output=True, text=True)
    try:
        s = json.loads(out.stdout)["streams"][0]
    except Exception:
        return None
    s["nb_frames"] = str(s.get("nb_frames"))
    return s


def main():
    files = sorted(glob.glob(DEMO + "/*.mp4"))
    print("checking %d files" % len(files))
    bad = []
    for f in files:
        s = probe(f)
        if s is None:
            bad.append((f, "probe failed"))
            continue
        diff = {k: (s.get(k), v) for k, v in WANT.items() if str(s.get(k)) != str(v)}
        if diff:
            bad.append((os.path.basename(f), diff))
    if bad:
        print("FORMAT MISMATCHES:")
        for f, d in bad:
            print("  ", f, d)
    else:
        print("all %d files match the reference format "
              "(h264/1920x1080/yuv420p/24fps/100 frames)" % len(files))

    # picks sheet: first clip of each source, plain variant, frame 40
    tiles = []
    for src in "ABCDE":
        p = "%s/%s00_demo2x2.mp4" % (DEMO, src)
        if not os.path.exists(p):
            continue
        cap = cv2.VideoCapture(p)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            continue
        fr = cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.rectangle(fr, (0, 0), (960, 34), (0, 0, 0), -1)
        cv2.putText(fr, "%s : %s" % (src, SRC[src]), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 120), 2)
        tiles.append(fr)
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
        sheet = np.vstack(rows)
        p = os.path.join(FIG, "picks_sheet_wild.png")
        cv2.imwrite(p, sheet)
        print("picks sheet ->", p)


if __name__ == "__main__":
    main()
