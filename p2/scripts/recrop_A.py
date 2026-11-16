"""Rebuild ONE clip set (A) with a motion-selected window, without touching anything else.

Why: the first A window was chosen by high-pass texture and landed on the building
facade at the top of the frame. It contained no moving subject at all, so it could not
exercise a temporal method -- the A results measured the wrong thing.

Window selection, corrected:
  1. estimate dense flow between sampled frames
  2. subtract the median flow vector (the dominant global camera translation)
  3. the window maximising the mean RESIDUAL flow is where things move independently
     of the background, i.e. where the pedestrians/cars are
Confirmed visually: candidate (0, 1168) frames the crosswalk, the pedestrian and the car.

Everything else (LR construction, sigma=1.5 + bicubic, 100 frames, 4 clips, GT window
1072x720 -> LR 268x180) is identical to the first run, so the numbers stay comparable.
"""
import cv2, numpy as np, os, json

SRC = "/tmp/VID_20261116_113830.mp4"
WIN = (1072, 720)
X0, Y0 = 0, 1168
NCLIP = 4
FRAMES = 100
ROOT = "/tmp/wild"


def down4(img):
    bl = cv2.GaussianBlur(img, (0, 0), 1.5)
    h, w = bl.shape[:2]
    return cv2.resize(bl, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)


def main():
    cap = cv2.VideoCapture(SRC)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    H, W = frames[0].shape[:2]
    tw, th = WIN
    print("A frames=%d frame %dx%d window at (%d,%d) %dx%d" % (len(frames), W, H, X0, Y0, tw, th))

    n = len(frames)
    m = NCLIP
    span = n - FRAMES
    starts = [int(round(20 + i * (span - 20) / max(1, m - 1))) for i in range(m)]
    info = []
    for ci, st in enumerate(starts):
        name = "A%02d" % ci
        gd = os.path.join(ROOT, "gt", name)
        ld = os.path.join(ROOT, "lr", name)
        os.makedirs(gd, exist_ok=True)
        os.makedirs(ld, exist_ok=True)
        for j in range(FRAMES):
            g = frames[st + j][Y0:Y0 + th, X0:X0 + tw]
            cv2.imwrite(os.path.join(gd, "%08d.png" % j), g)
            cv2.imwrite(os.path.join(ld, "%08d.png" % j), down4(g))
        info.append({"clip": name, "start_frame": st, "n": FRAMES})
        print("  clip %s frames %d..%d GT %dx%d LR %dx%d"
              % (name, st, st + FRAMES - 1, tw, th, tw // 4, th // 4), flush=True)

    p = os.path.join(ROOT, "prep.json")
    rep = json.load(open(p)) if os.path.exists(p) else {}
    rep.setdefault("A", {})["window"] = [X0, Y0, tw, th]
    rep["A"]["clips"] = info
    rep["A"]["window_selection"] = "median-removed residual flow (motion), not texture"
    json.dump(rep, open(p, "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
