#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tasks 7 + 16 -- qualitative evidence for the flow-aligned variants.

What the checklist asks for and how it is answered here:

  * one fixed clip + frame + ROI showing ``GT / Bicubic / EDSR / tavg5 /
    flowtavg5 / flowtavg5+USM`` together (task 7) -- the Part 1 figures never had
    the flow variants in them, because those predictions are produced in the same
    pass as their metrics and were discarded;
  * at least three ROI kinds: text/thin-line, texture, motion edge;
  * consecutive 3-5 frames of the SAME ROI, so ghosting / smearing / ringing can
    be judged (a static patch cannot show temporal behaviour);
  * Part 2 panel completion ``GT / Bicubic / EDSR / EDVR / TecoGAN`` (task 16) --
    EDVR was missing from some patches;
  * the deliverable videos keep the source FPS (24), replacing the old fixed 10 fps.

Nothing is stored beyond the crops and the montages: predictions are computed for
one clip at a time, cropped down to the ROIs, and dropped.  8 methods x 100 x 720p
would be 5 GB if kept.

ROI choice is scripted and fixed, not eyeballed per figure:
  text      highest mean |Laplacian| inside a window on the GT frame (thin strokes)
  texture   highest local variance in the GT frame that is NOT the text window
  motion    highest mean |flow| magnitude (bicubic -> next frame) location
The three windows are then used unchanged for every method and every frame.

Usage:
    /tmp/srenv/bin/python ~/p0/scripts/qualitative.py --out ~/p0/out/qual --clip 004
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import types

os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import numpy as np
import cv2
import torch
import torchvision

_m = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F  # noqa: E402

for _n in dir(_F):
    setattr(_m, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _m
torch.backends.cudnn.benchmark = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from evalkit import data as D  # noqa: E402

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
TECO_DIR = "/tmp/p2/tecogan"
EDVR_DIR = "/tmp/p2/edvr_frame"
CK_EDSR = ("/data/disk1/shared/aiaa3201/checkpoints/EDSR_baseline_x4/"
           "EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth")
CK_EDVR = ("/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
           "EDVR_L_x4_SR_REDS_official-9f5f5039.pth")

ROI_W, ROI_H, ZOOM = 160, 90, 4


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def unsharp(f, radius=1.0, amount=1.5):
    ff = f.astype(np.float32)
    return np.clip(np.rint(ff + amount * (ff - cv2.GaussianBlur(ff, (0, 0), radius))),
                   0, 255).astype(np.uint8)


def naive(frames, kernel):
    wt = np.asarray(kernel, np.float32)
    wt /= wt.sum()
    k = len(wt) // 2
    n = len(frames)
    h, w = frames[0].shape[:2]
    out = np.empty((n, h, w, 3), np.uint8)
    for t in range(n):
        acc = np.zeros((h, w, 3), np.float32)
        for d, wi in enumerate(wt):
            j = min(max(t + d - k, 0), n - 1)
            acc += wi * frames[j].astype(np.float32)
        out[t] = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
    return out


def fb(t, s, scale=4):
    h, w = t.shape
    ts = cv2.resize(t, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    ss = cv2.resize(s, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(ts, ss, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * float(scale)


def flow_avg(frames, grays, kernel, gx, gy):
    wt = np.asarray(kernel, np.float32)
    wt /= wt.sum()
    k = len(wt) // 2
    n = len(frames)
    h, w = frames[0].shape[:2]
    out = np.empty((n, h, w, 3), np.uint8)
    for t in range(n):
        acc = np.zeros((h, w, 3), np.float32)
        for d, wi in enumerate(wt):
            j = min(max(t + d - k, 0), n - 1)
            if j == t:
                acc += wi * frames[t].astype(np.float32)
            else:
                f = fb(grays[t], grays[j])
                acc += wi * cv2.remap(frames[j], gx + f[..., 0], gy + f[..., 1],
                                      cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        out[t] = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
    return out


class EDSR:
    def __init__(self, device="cuda"):
        from basicsr.archs.edsr_arch import EDSR as Net
        ck = torch.load(CK_EDSR, map_location="cpu", weights_only=False)["params"]
        net = Net(3, 3, 64, 16, 4)
        net.load_state_dict(ck, strict=True)
        self.net = net.to(device).eval()
        for q in self.net.parameters():
            q.requires_grad_(False)
        self.device = device

    def __call__(self, lr, bs=4):
        out = []
        for i in range(0, len(lr), bs):
            x = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float().div_(255.0)
                             for f in lr[i:i + bs]]).to(self.device)
            with torch.no_grad():
                y = self.net(x)
            out.append(y.clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1)
                       .cpu().numpy())
        return np.concatenate(out, 0)


# ------------------------------------------------------------------ ROI selection
def pick_rois(gt, nxt, grays, grays_next):
    """Three fixed windows: text, texture, motion. Deterministic, no eyeballing."""
    g = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    var = cv2.GaussianBlur((g - cv2.GaussianBlur(g, (0, 0), 3)) ** 2, (0, 0), 9)
    f = fb(grays, grays_next)
    mag = cv2.GaussianBlur(np.sqrt((f ** 2).sum(-1)), (0, 0), 9)

    def best(score, avoid=None, min_gap=0):
        s = cv2.integral(score)
        H, W = score.shape
        cand, bxy = -1e9, (W // 2 - ROI_W // 2, H // 2 - ROI_H // 2)
        for y in range(0, H - ROI_H, 8):
            for x in range(0, W - ROI_W, 8):
                if avoid is not None:
                    ax, ay = avoid
                    if (abs(x - ax) < min_gap + ROI_W and abs(y - ay) < min_gap + ROI_H):
                        continue
                tot = (s[y + ROI_H, x + ROI_W] - s[y, x + ROI_W]
                       - s[y + ROI_H, x] + s[y, x])
                if tot > cand:
                    cand, bxy = tot, (x, y)
        return bxy

    text_xy = best(lap)
    tex_xy = best(var, avoid=text_xy, min_gap=ROI_W)
    mot_xy = best(mag, avoid=text_xy, min_gap=ROI_W)
    return {"text": text_xy, "texture": tex_xy, "motion": mot_xy}


def crop(img, xy):
    x, y = xy
    c = img[y:y + ROI_H, x:x + ROI_W]
    return cv2.resize(c, (ROI_W * ZOOM, ROI_H * ZOOM),
                      interpolation=cv2.INTER_NEAREST)


def label(img, txt, scale=0.5):
    out = img.copy()
    cv2.putText(out, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(out, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def montage(rows, names, title=None):
    """rows = list of frame-index groups; each group is one horizontal strip."""
    strips = []
    for r in rows:
        strips.append(np.hstack([label(im, nm) for im, nm in zip(r, names)]))
    H = sum(s.shape[0] for s in strips)
    W = max(s.shape[1] for s in strips)
    canvas = np.zeros((H, W, 3), np.uint8)
    y = 0
    for s in strips:
        canvas[y:y + s.shape[0], :s.shape[1]] = s
        y += s.shape[0]
    if title:
        bar = np.zeros((26, W, 3), np.uint8)
        cv2.putText(bar, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (235, 235, 235), 1, cv2.LINE_AA)
        canvas = np.vstack([bar, canvas])
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", default="011")
    ap.add_argument("--roi", nargs="*", default=None,
                    help="override ROI windows, e.g. text=904,288 texture=176,56")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--consecutive", type=int, default=4)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--video", action="store_true",
                    help="also render a 6-up comparison video at the source fps")
    args = ap.parse_args()
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out_dir, exist_ok=True)

    clip = args.clip
    idx = D.index_dir(os.path.join(GT_ROOT, clip))
    ids = sorted(idx)[:args.frames]
    gt = [rd(idx[i]) for i in ids]
    lidx = D.index_dir(os.path.join(LR_ROOT, clip))
    lr = [rd(lidx[i]) for i in ids]
    H, W = gt[0].shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    bic = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr]
    grays = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in bic]

    t0 = time.time()
    edsr = EDSR()(lr)
    tavg5 = naive(bic, (1, 2, 3, 2, 1))
    f5 = flow_avg(bic, grays, (1, 2, 3, 2, 1), gx, gy)
    f5u = np.stack([unsharp(x) for x in f5])
    print("predictions ready in %.1fs" % (time.time() - t0), flush=True)

    part2 = {}
    ed = D.index_dir(os.path.join(EDVR_DIR, clip)) if os.path.isdir(
        os.path.join(EDVR_DIR, clip)) else None
    te = D.index_dir(os.path.join(TECO_DIR, clip)) if os.path.isdir(
        os.path.join(TECO_DIR, clip)) else None
    if ed:
        part2["edvr"] = [rd(ed[i]) for i in ids]
    if te:
        part2["tecogan"] = [rd(te[i]) for i in ids]

    # ---- ROI choice on a middle frame, then frozen for all frames/methods
    mid = min(40, len(ids) - 2)
    rois = pick_rois(gt[mid], gt[mid + 1], grays[mid], grays[mid + 1])
    roi_source = "scripted"
    if args.roi:
        for item in args.roi:
            k, v = item.split("=")
            if k not in rois:
                raise SystemExit("unknown ROI %r (have %s)" % (k, list(rois)))
            x, y = (int(t) for t in v.split(","))
            rois[k] = (x, y)
        roi_source = "manual override: " + " ".join(args.roi)
    print("ROIs (x,y) on clip %s frame %d: %s [%s]"
          % (clip, ids[mid], rois, roi_source), flush=True)

    methods1 = {"GT": gt, "Bicubic": bic, "EDSR": edsr, "tavg5": tavg5,
                "flowtavg5": f5, "flowtavg5+USM": f5u}
    for kind, xy in rois.items():
        strip = [crop(gt[mid], xy)]
        for nm, arr in methods1.items():
            if nm == "GT":
                continue
            strip.append(crop(arr[mid], xy))
        m = montage([strip], list(methods1.keys()),
                    "%s  clip %s  frame id %d  ROI=%s (x=%d,y=%d, %dx%d, x%d)"
                    % ("Part 1 flow ablation", clip, ids[mid], kind, xy[0], xy[1],
                       ROI_W, ROI_H, ZOOM))
        cv2.imwrite(os.path.join(out_dir, "p1_%s_%s_detail.png" % (kind, clip)),
                    cv2.cvtColor(m, cv2.COLOR_RGB2BGR))

        # consecutive frames: same ROI, one row per frame
        rows, names = [], list(methods1.keys())
        for k in range(args.consecutive):
            fi = min(mid + k, len(ids) - 1)
            row = [crop(gt[fi], xy)] + [crop(arr[fi], xy) for nm, arr in methods1.items()
                                        if nm != "GT"]
            rows.append(row)
        m2 = montage(rows, names,
                     "%s  clip %s  frames %d-%d  ROI=%s (rows are consecutive frames)"
                     % ("Part 1 consecutive", clip, ids[mid],
                        ids[min(mid + args.consecutive - 1, len(ids) - 1)], kind))
        cv2.imwrite(os.path.join(out_dir, "p1_%s_%s_consecutive.png" % (kind, clip)),
                    cv2.cvtColor(m2, cv2.COLOR_RGB2BGR))

        if part2:
            names2 = ["GT", "Bicubic", "EDSR"] + [k.upper() if k == "edvr" else "TecoGAN"
                                                 for k in ("edvr", "tecogan") if k in part2]
            strip2 = [crop(gt[mid], xy), crop(bic[mid], xy), crop(edsr[mid], xy)]
            for k in ("edvr", "tecogan"):
                if k in part2:
                    strip2.append(crop(part2[k][mid], xy))
            m3 = montage([strip2], names2,
                         "%s  clip %s  frame id %d  ROI=%s"
                         % ("Part 2 five-way panel", clip, ids[mid], kind))
            cv2.imwrite(os.path.join(out_dir, "p2_%s_%s_panel.png" % (kind, clip)),
                        cv2.cvtColor(m3, cv2.COLOR_RGB2BGR))

    # ---- full-frame + zoom "hero" image for the report front page
    hero = np.hstack([bic[mid], f5u[mid]])
    cv2.imwrite(os.path.join(out_dir, "hero_%s_%d.png" % (clip, ids[mid])),
                cv2.cvtColor(hero, cv2.COLOR_RGB2BGR))

    # ---- deliverable video: 6 panels, SOURCE fps (task 16: not the old 10 fps)
    if args.video:
        from imageio_ffmpeg import get_ffmpeg_exe
        panes = [("GT", gt), ("Bicubic", bic), ("EDSR", edsr), ("tavg5", tavg5),
                 ("flowtavg5", f5), ("flowtavg5+USM", f5u)]
        PW, PH = 640, 360
        out = os.path.join(out_dir, "%s_6up_f%dfps.mp4" % (clip, args.fps))
        cmd = [get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (PW * 2, PH * 3),
               "-r", str(args.fps), "-i", "-", "-an",
               "-c:v", "libx264", "-preset", "slow", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for t in range(len(ids)):
            tiles = []
            for nm, arr in panes:
                f = cv2.resize(arr[t], (PW, PH), interpolation=cv2.INTER_AREA)
                f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                cv2.putText(f, "%s  clip %s f%03d" % (nm, clip, ids[t]), (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(f, "%s  clip %s f%03d" % (nm, clip, ids[t]), (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(f)
            grid = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4]),
                              np.hstack(tiles[4:6])])
            p.stdin.write(grid.tobytes())
        p.stdin.close()
        err = p.stderr.read().decode("utf-8", "replace")
        if p.wait() != 0:
            print("ffmpeg failed:\n" + err[-800:])
        else:
            print("video:", out, flush=True)

    json.dump({"clip": clip, "frame_id": ids[mid], "rois": {k: list(v)
                                                           for k, v in rois.items()},
               "roi_size": [ROI_W, ROI_H], "zoom": ZOOM,
               "fps": args.fps, "consecutive": args.consecutive,
               "methods_part1": list(methods1),
               "methods_part2": list(part2),
               "roi_source": roi_source,
               "roi_rule": "scripted: text = max mean|Laplacian| on the GT frame; "
                           "texture = max local variance avoiding the text window; "
                           "motion = max smoothed |flow| avoiding the text window. "
                           "NOTE the heuristic finds 'busiest detail', not semantic "
                           "text; where the clip has no signage the label is the "
                           "window rule, not a claim about content."},
              open(os.path.join(out_dir, "qual_%s.json" % clip), "w", encoding="utf-8"),
              indent=1)
    print("->", out_dir)


if __name__ == "__main__":
    main()
