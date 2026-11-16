#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1 Part 2 demo videos -- 2x2 grid, 1920x1080, H.264, presentation-ready.

    +---------------------+---------------------+
    |        GT           |      Bicubic        |
    +---------------------+---------------------+
    |       EDVR          |      TecoGAN        |
    +---------------------+---------------------+

Two variants per clip:

  A) plain    -- full 720p frame, one panel per method (960x540 each)
  B) zoominset-- same, plus a 2x magnified loupe in the corner and the source
                 rectangle drawn on every panel, so the texture difference is
                 readable even from the back of a lecture room.

Fixes over the 2026-11-14 version (which crashed on clip 013 with
``cv2.error: (-215:Assertion failed) !ssize.empty()`` in ``loupe``):

  * ``loupe`` clamps against the *actual* image size and falls back to a
    centre crop when the requested window cannot hold a CWxCH patch, so an
    out-of-range coordinate can never produce an empty slice again.
  * ``busy_box`` resizes GT/bicubic to a common size and refuses to search when
    the frame is smaller than the window (returns the centre instead).
  * frames are matched by parsed **frame id**, not by position in a sorted glob,
    so a gap in one method's output can no longer shift the whole video.
  * a missing prediction panel is rendered as a labelled placeholder instead of
    crashing the render.
  * ffmpeg's stderr is captured and reported; a failed encode deletes the
    partial file and exits non-zero.

Usage:
  python p2_demo_2x2.py [--inset] [--fps N] [--out DIR] [--preset P] clip [clip...]
  python p2_demo_2x2.py --all-4stream            # every clip that has GT+bic+EDVR+TecoGAN
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, re, glob, subprocess
import numpy as np
import cv2

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
TE = "/tmp/p2/tecogan"
ED = "/tmp/p2/edvr_frame"
OUT = os.environ.get("DEMO_OUT", "/tmp/p2/demo")
os.makedirs(OUT, exist_ok=True)

PW, PH = 960, 540           # panel size -> 1920x1080 total
FPS = 24                    # REDS native frame rate
FONT = cv2.FONT_HERSHEY_SIMPLEX

# loupe: 192x108 crop blown up 2x
CW, CH, CSC = 192, 108, 2
ZP = (PW - CW * CSC - 14, PH - CH * CSC - 14)      # loupe top-left, bottom-right corner

PALETTE = {
    "GT":      (120, 255, 120),
    "Bicubic": (90, 210, 255),
    "EDVR":    (120, 255, 200),
    "TecoGAN": (200, 190, 255),
}

_DIG = re.compile(r"(\d+)")


def ffmpeg_exe():
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def frame_id(name):
    m = _DIG.findall(os.path.basename(name))
    return int(m[-1]) if m else None


def index_dir(path):
    """{frame_id: path}; first file wins and duplicates are reported."""
    out, dup = {}, []
    for f in sorted(glob.glob(os.path.join(path, "*.png"))):
        i = frame_id(f)
        if i is None:
            continue
        if i in out:
            dup.append(os.path.basename(f))
        out[i] = f
    if dup:
        print("  warn: duplicate frame ids in %s: %s" % (path, dup[:5]))
    return out


def load(p, size=(PW, PH)):
    a = cv2.imread(p)
    if a is None:
        return None
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    elif a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    if size and (a.shape[1], a.shape[0]) != tuple(size):
        a = cv2.resize(a, tuple(size), interpolation=cv2.INTER_AREA)
    return a


def put(img, text, org, size, color, thick=2):
    cv2.putText(img, text, org, FONT, size, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, size, color, thick, cv2.LINE_AA)


def busy_box(gt_p, bic_p):
    """Window where bicubic loses the most texture vs GT.

    Maximising mean|GT - bicubic| inside the window finds exactly the patches a
    presenter wants to point at: fine detail that survives in GT and is smeared
    away by upsampling. Purely high-gradient areas are rejected implicitly,
    because a region that is already soft in GT yields no difference.
    """
    H, W = gt_p.shape[:2]
    cx, cy = max(0, W // 2 - CW // 2), max(0, H // 2 - CH // 2)
    if H < CH + 2 or W < CW + 2:
        return (cx, cy)                      # tiny frame: centre, never negative
    if bic_p.shape[:2] != (H, W):
        bic_p = cv2.resize(bic_p, (W, H), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(gt_p, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(bic_p, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = np.abs(g - b)
    s = cv2.integral(d)
    best, bxy = -1.0, (cx, cy)
    ys = list(range(0, H - CH, 16))
    xs = list(range(0, W - CW, 16))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    for y in ys:
        for x in xs:
            y2, x2 = min(y + CH, H), min(x + CW, W)
            tot = s[y2, x2] - s[y, x2] - s[y2, x] + s[y, x]
            if tot > best:
                best, bxy = tot, (x, y)
    return bxy


def loupe(img_full, xy):
    """Magnified crop of the already-panel-sized image.

    The 2026-11-14 version assumed ``img_full`` was exactly PHxPW and could slice
    an empty array (clip 013 crash). Everything here is clamped against the real
    shape, with a centre-crop fallback if even that cannot hold a CWxCH patch.
    """
    H, W = img_full.shape[:2]
    cw, ch = min(CW, W), min(CH, H)
    x, y = int(round(xy[0])), int(round(xy[1]))
    x = max(0, min(x, W - cw))
    y = max(0, min(y, H - ch))
    c = img_full[y:y + ch, x:x + cw]
    if c.size == 0 or c.shape[0] == 0 or c.shape[1] == 0:
        x, y = max(0, (W - cw) // 2), max(0, (H - ch) // 2)
        c = img_full[y:y + ch, x:x + cw]
    if c.size == 0:
        raise RuntimeError("loupe: cannot crop any patch from a %dx%d image" % (W, H))
    if (c.shape[1], c.shape[0]) != (CW, CH):
        c = cv2.resize(c, (CW, CH), interpolation=cv2.INTER_AREA)
    return cv2.resize(c, (CW * CSC, CH * CSC), interpolation=cv2.INTER_NEAREST), (x, y)


def panel(img, name, header, xy=None, inset=False):
    if img is None:
        img = np.full((PH, PW, 3), 24, np.uint8)
        put(img, "n/a", (PW // 2 - 40, PH // 2), 1.2, (0, 0, 255))
    if img.shape[:2] != (PH, PW):
        img = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_AREA)
    t = img.copy()
    put(t, name, (14, 38), 0.95, PALETTE.get(name, (255, 255, 255)))
    if inset and xy is not None:
        lou, (x, y) = loupe(img, xy)
        cv2.rectangle(t, (x, y), (x + min(CW, img.shape[1]), y + min(CH, img.shape[0])),
                      (60, 255, 60), 2, cv2.LINE_AA)
        px, py = ZP
        h, w = lou.shape[:2]
        t[py:py + h, px:px + w] = lou
        cv2.rectangle(t, (px, py), (px + w, py + h), (60, 255, 60), 2)
    put(t, header, (14, PH - 16), 0.55, (235, 235, 235), 1)
    return t


def grid(gt, bic, edvr, teco, header, xy=None, inset=False):
    tl = panel(gt, "GT", header, xy, inset)
    tr = panel(bic, "Bicubic", header, xy, inset)
    bl = panel(edvr, "EDVR", header, xy, inset)
    br = panel(teco, "TecoGAN", header, xy, inset)
    g = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])
    cv2.line(g, (PW, 0), (PW, 2 * PH), (10, 10, 10), 3)
    cv2.line(g, (0, PH), (2 * PW, PH), (10, 10, 10), 3)
    return g


def streams(clip):
    """Matched lists of (frame_id, gt_path, lr_path, edvr_path, tecogan_path)."""
    gt = index_dir(os.path.join(GT_ROOT, clip))
    lr = index_dir(os.path.join(LR_ROOT, clip))
    ed = index_dir(os.path.join(ED, clip))
    te = index_dir(os.path.join(TE, clip))
    ids = sorted(set(gt) & set(lr) & set(ed) & set(te))
    if not ids:
        return []
    missing = sorted(set(gt) - set(ids))
    if missing:
        print("  %s: %d GT frames without all four streams (e.g. %s) -- skipped"
              % (clip, len(missing), missing[:3]))
    return [(i, gt[i], lr[i], ed[i], te[i]) for i in ids]


def make_video(clip, inset=False, fps=FPS, preset="slow"):
    rows = streams(clip)
    if not rows:
        print("skip %s: no clip with all four streams" % clip)
        return None
    n = len(rows)

    # loupe window chosen on a middle frame where every stream exists
    j = min(40, n - 1)
    _gt = load(rows[j][1])
    _lr = load(rows[j][2], None)
    _bic = cv2.resize(_lr, (PW, PH), interpolation=cv2.INTER_CUBIC) if _lr is not None else None
    xy = busy_box(_gt, _bic) if (_gt is not None and _bic is not None) else (PW // 2, PH // 2)

    suffix = "demo2x2_inset" if inset else "demo2x2"
    out = os.path.join(OUT, "%s_%s.mp4" % (clip, suffix))
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (PW * 2, PH * 2),
           "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", preset, "-crf", "18",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for k, (fid, gtp, lrp, edp, tep) in enumerate(rows):
            gt = load(gtp)
            lr = load(lrp, None)
            bic = cv2.resize(lr, (PW, PH), interpolation=cv2.INTER_CUBIC) \
                if lr is not None else None
            edv = load(edp)
            teo = load(tep)
            hdr = "clip %s   frame %03d/%03d   (id %d)" % (clip, k + 1, n, fid)
            p.stdin.write(grid(gt, bic, edv, teo, hdr, xy, inset).tobytes())
    finally:
        try:
            p.stdin.close()
        except Exception:
            pass
        err = p.stderr.read().decode("utf-8", "replace")
        rc = p.wait()
    if rc != 0:
        print("ffmpeg failed for %s (rc=%d):\n%s" % (clip, rc, err[-800:]))
        if os.path.exists(out):
            os.remove(out)
        return None
    return out


def contact_sheet(clips, frame=40, inset=False):
    tiles = []
    for c in clips:
        rows = streams(c)
        if not rows:
            continue
        i = min(frame, len(rows) - 1)
        fid, gtp, lrp, edp, tep = rows[i]
        gt = load(gtp)
        lr = load(lrp, None)
        bic = cv2.resize(lr, (PW, PH), interpolation=cv2.INTER_CUBIC) if lr is not None else None
        xy = busy_box(gt, bic) if (gt is not None and bic is not None) else (PW // 2, PH // 2)
        tiles.append(grid(gt, bic, load(edp), load(tep),
                          "clip %s  frame %d" % (c, fid), xy, inset))
    if not tiles:
        return None
    rows = []
    for k in range(0, len(tiles), 2):
        chunk = tiles[k:k + 2]
        if len(chunk) == 1:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(chunk))
    img = np.vstack(rows)
    scale = 0.5 if img.shape[1] > 2400 else 1.0
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    name = "contact_sheet_inset.png" if inset else "contact_sheet.png"
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    return path


def clips_with_four_streams():
    out = []
    for c in sorted(os.listdir(GT_ROOT)):
        if not os.path.isdir(os.path.join(GT_ROOT, c)):
            continue
        if os.path.isdir(os.path.join(ED, c)) and os.path.isdir(os.path.join(TE, c)):
            if streams(c):
                out.append(c)
    return out


if __name__ == "__main__":
    argv = sys.argv[1:]
    inset = "--inset" in argv
    preset = "slow"
    fps = FPS
    if "--preset" in argv:
        preset = argv[argv.index("--preset") + 1]
    if "--fps" in argv:
        fps = int(argv[argv.index("--fps") + 1])
    if "--out" in argv:
        OUT = argv[argv.index("--out") + 1]
        os.makedirs(OUT, exist_ok=True)
    if "--all-4stream" in argv:
        clips = clips_with_four_streams()
        print("clips with all four streams:", " ".join(clips), flush=True)
    else:
        skip = {"--preset", "--fps", "--out"}
        clips, it = [], iter(argv)
        for a in it:
            if a in skip:
                next(it, None)
                continue
            if not a.startswith("-"):
                clips.append(a)
        clips = clips or ["003", "029", "002", "001"]
    made = []
    for c in clips:
        v = make_video(c, inset, fps=fps, preset=preset)
        print("%s -> %s" % (c, v), flush=True)
        if v:
            made.append(c)
    print("contact:", contact_sheet(made, inset=inset), flush=True)
