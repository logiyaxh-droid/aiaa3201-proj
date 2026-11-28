"""Build GT/LR pairs for a wild clip set, with automatic motion-based window selection.

WHY THE WINDOW IS AUTOMATIC
  The first iteration picked the crop window by texture (high-pass std) and landed on a
  building facade with no moving subject, which invalidated the whole clip. Texture is the
  wrong signal: a temporal method needs *independent* motion. A raw frame difference is
  also wrong, because camera drift makes the whole frame differ. The signal that works is
  the residual of the optical flow after removing its median (the dominant global
  translation).

  For handheld footage that walks forward, the residual is large everywhere (radial
  expansion), so the search degenerates to "anywhere". In that case the window is placed
  at the frame centre, which is where the operator is pointing, and the residual score is
  reported so the degeneration is visible rather than silent.

WHY THE WINDOW IS FIXED AT 1280x720
  Every source must produce LR 320x180 so the results are comparable to the REDS table and
  to the previously built A/B sets. Taking the largest possible crop would give LR 476x268
  -- a materially easier task, and the numbers would no longer sit in the same table.

PROTOCOL (state this verbatim in the report)
  * The phone recording is GT; wild footage has no other ground truth.
  * LR = 4x downsample, Gaussian sigma=1.5 + INTER_CUBIC. Calibrated against the course's
    official REDS LR: 42.38 dB agreement, vs 27.17 dB for plain INTER_CUBIC.
  * Therefore: real content, synthetic bicubic degradation. Only the content is real.

usage: python prep_auto.py <video> <tag> [nclip]
"""
import cv2, numpy as np, os, json, sys

ROOT = "/tmp/wild"
FRAMES = 100
TW, TH = 1280, 720


def pick_window(frames, tw, th, ds=4):
    """Window maximising mean residual flow (motion independent of the camera)."""
    small = []
    for fr in frames:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        small.append(cv2.resize(g, (g.shape[1] // ds, g.shape[0] // ds), interpolation=cv2.INTER_AREA))
    H, W = small[0].shape
    acc = np.zeros((H, W), np.float32)
    n = 0
    for a, b in zip(small[:-1], small[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 5, 7, 1.5, 0)
        med = np.median(f.reshape(-1, 2), axis=0)
        res = f - med.reshape(1, 1, 2)
        acc += np.hypot(res[..., 0], res[..., 1])
        n += 1
    acc /= max(n, 1)
    tws, ths = tw // ds, th // ds
    if ths > H or tws > W:
        raise RuntimeError("window %dx%d too big at 1/%d for frame %dx%d"
                           % (tw, th, ds, W, H))
    best, bxy = -1.0, (0, 0)
    for y in range(0, H - ths + 1, 4):
        for x in range(0, W - tws + 1, 4):
            m = float(acc[y:y + ths, x:x + tws].mean())
            if m > best:
                best, bxy = m, (x * ds, y * ds)
    # degeneracy check: if the best window is barely better than the frame average the
    # residual map is flat (global expansion) and the argmax carries no information
    frame_mean = float(acc.mean())
    if frame_mean > 0 and best / frame_mean < 1.05:
        bxy = ((W - tws) // 2, (H - ths) // 2)
        return bxy[0] * ds, bxy[1] * ds, best, True
    return bxy[0], bxy[1], best, False


def down4(img):
    bl = cv2.GaussianBlur(img, (0, 0), 1.5)
    h, w = bl.shape[:2]
    return cv2.resize(bl, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)


def main():
    path, tag = sys.argv[1], sys.argv[2]
    nclip = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    H, W = frames[0].shape[:2]
    tw, th = TW, TH
    if th > H or tw > W:
        raise RuntimeError("%s is %dx%d, smaller than the 1280x720 window" % (tag, W, H))

    step = max(1, len(frames) // 40)
    x0, y0, score, flat = pick_window(frames[::step], tw, th)
    print("%s: %d frames %dx%d -> window %dx%d at (%d,%d) residual=%.3f%s"
          % (tag, len(frames), W, H, tw, th, x0, y0, score,
             "  [FLAT: residual map degenerate, centred]" if flat else ""), flush=True)
    cv2.imwrite("%s/preview_%s.png" % (ROOT, tag),
                frames[min(30, len(frames) - 1)][y0:y0 + th, x0:x0 + tw])

    n = len(frames)
    span = n - FRAMES
    if span < 20:
        raise RuntimeError("%s has only %d frames, cannot cut a %d-frame clip" % (tag, n, FRAMES))
    m = nclip
    starts = [int(round(20 + i * (span - 20) / max(1, m - 1))) for i in range(m)]
    info = []
    for ci, st in enumerate(starts):
        name = "%s%02d" % (tag, ci)
        gd = os.path.join(ROOT, "gt", name)
        ld = os.path.join(ROOT, "lr", name)
        os.makedirs(gd, exist_ok=True)
        os.makedirs(ld, exist_ok=True)
        for j in range(FRAMES):
            g = frames[st + j][y0:y0 + th, x0:x0 + tw]
            cv2.imwrite(os.path.join(gd, "%08d.png" % j), g)
            d = down4(g)
            assert d.shape[0] % 4 == 0 and d.shape[1] % 4 == 0, d.shape
            cv2.imwrite(os.path.join(ld, "%08d.png" % j), d)
        info.append({"clip": name, "start_frame": st, "n": FRAMES})
        print("  %s frames %d..%d GT %dx%d LR %dx%d"
              % (name, st, st + FRAMES - 1, tw, th, tw // 4, th // 4), flush=True)

    p = os.path.join(ROOT, "prep.json")
    rep = json.load(open(p)) if os.path.exists(p) else {}
    rep.setdefault(tag, {})
    rep[tag]["src"] = path
    rep[tag]["frame_wh"] = [W, H]
    rep[tag]["window"] = [x0, y0, tw, th]
    rep[tag]["clips"] = info
    rep[tag]["window_selection"] = "median-removed residual flow, centred if degenerate"
    rep[tag]["residual_score"] = score
    rep[tag]["window_degenerate"] = flat
    json.dump(rep, open(p, "w"), indent=1)
    print("done", tag)


if __name__ == "__main__":
    main()
