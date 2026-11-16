#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wild-video demo videos in exactly the format of p1_results/p2/demo.

Format copied from p2_demo_2x2.py (do not change it, the user asked for a match):
    1920x1080, 24 fps, H.264 yuv420p + faststart, CRF 18
    2x2 grid of 960x540 panels:  GT | Bicubic
                                 EDVR | TecoGAN
    method name top-left of each panel, "clip X  frame k/n  (id i)" along the bottom
    optional _inset variant: 2x loupe bottom-right + green source rectangle

WHY THIS DOES NOT REUSE p2_demo_2x2.py
  1. That script reads PNG sequences for all four streams; we only have EDVR as a single
     sample frame per wild clip, because storing 20 clips x 100 frames at 720p would need
     ~2.8 GB and the box has 4.1 GB free. So EDVR is computed here and piped straight into
     ffmpeg, never touching disk.
  2. The LR frames for A/B were removed during the last run; regenerating LR from GT on the
     fly (sigma=1.5 + INTER_CUBIC, 4x) keeps every source on the identical path.

GPU loop for one clip: all 100 EDVR outputs are held in RAM (~276 MB), encoded, released.
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, re, glob, types, subprocess, json
import numpy as np
import cv2
import torch
import torchvision

_mod = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as _F
for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = _mod

from basicsr.archs.edvr_arch import EDVR

torch.backends.cudnn.benchmark = True

WILD = "/tmp/wild"
HOLD = WILD + "/_ab_hold"
CK = ("/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
      "EDVR_L_x4_SR_REDS_official-9f5f5039.pth")
DEV = "cuda"

PW, PH = 960, 540
FPS = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
CW, CH, CSC = 192, 108, 2
ZP = (PW - CW * CSC - 14, PH - CH * CSC - 14)

PALETTE = {
    "GT":      (120, 255, 120),
    "Bicubic": (90, 210, 255),
    "EDVR":    (120, 255, 200),
    "TecoGAN": (200, 190, 255),
}


def ffmpeg_exe():
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def gt_dir(clip):
    p = os.path.join(WILD, "gt", clip)
    if os.path.isdir(p) and glob.glob(p + "/*.png"):
        return p
    p = os.path.join(HOLD, "gt_" + clip)
    return p if os.path.isdir(p) else None


def down4(img):
    bl = cv2.GaussianBlur(img, (0, 0), 1.5)
    h, w = bl.shape[:2]
    return cv2.resize(bl, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)


def put(img, text, org, size, color, thick=2):
    cv2.putText(img, text, org, FONT, size, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, size, color, thick, cv2.LINE_AA)


def busy_box(gt_p, bic_p):
    H, W = gt_p.shape[:2]
    cx, cy = max(0, W // 2 - CW // 2), max(0, H // 2 - CH // 2)
    if H < CH + 2 or W < CW + 2:
        return (cx, cy)
    if bic_p.shape[:2] != (H, W):
        bic_p = cv2.resize(bic_p, (W, H), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(gt_p, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(bic_p, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = np.abs(g - b)
    s = cv2.integral(d)
    best, bxy = -1.0, (cx, cy)
    for y in range(0, max(1, H - CH), 16):
        for x in range(0, max(1, W - CW), 16):
            y2, x2 = min(y + CH, H), min(x + CW, W)
            tot = s[y2, x2] - s[y, x2] - s[y2, x] + s[y, x]
            if tot > best:
                best, bxy = tot, (x, y)
    return bxy


def loupe(img_full, xy):
    H, W = img_full.shape[:2]
    cw, ch = min(CW, W), min(CH, H)
    x, y = int(round(xy[0])), int(round(xy[1]))
    x = max(0, min(x, W - cw))
    y = max(0, min(y, H - ch))
    c = img_full[y:y + ch, x:x + cw]
    if c.size == 0:
        x, y = max(0, (W - cw) // 2), max(0, (H - ch) // 2)
        c = img_full[y:y + ch, x:x + cw]
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


# ------------------------------------------------------------------ EDVR
_ck = torch.load(CK, map_location="cpu", weights_only=False)["params"]
model = EDVR(num_in_ch=3, num_out_ch=3, num_feat=128, num_frame=5,
             deformable_groups=8, num_extract_block=5, num_reconstruct_block=40,
             center_frame_idx=2, hr_in=False, with_predeblur=False, with_tsa=True)
model.load_state_dict(_ck, strict=True)
model = model.cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)


def rd(path):
    a = np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def edvr_seq(lr_frames, n):
    """All n EDVR outputs as BGR uint8, sliding window with edge clamping."""
    out = []
    for c in range(n):
        win = [max(0, min(n - 1, c + d - 2)) for d in range(5)]
        x = torch.stack([lr_frames[w] for w in win]).unsqueeze(0).cuda()
        with torch.no_grad():
            y = model(x)
        y = y[0].clamp_(0, 1).mul_(255).round_().byte().permute(1, 2, 0).cpu().numpy()
        out.append(cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
    return out


def main():
    clips = sys.argv[1:]
    outdir = os.environ.get("DEMO_OUT", "/tmp/wild/demo")
    os.makedirs(outdir, exist_ok=True)
    only_inset = os.environ.get("ONLY_INSET") == "1"

    for clip in clips:
        gd = gt_dir(clip)
        te = os.path.join(WILD, "part2", clip)
        if gd is None:
            print("skip %s: no GT" % clip)
            continue
        gtf = sorted(glob.glob(gd + "/*.png"))
        tef = sorted(glob.glob(te + "/*.png"))
        tef = [f for f in tef if os.path.basename(f)[0].isdigit()]
        n = min(len(gtf), len(tef))
        if n < 10:
            print("skip %s: GT %d tecogan %d" % (clip, len(gtf), len(tef)))
            continue
        gtf, tef = gtf[:n], tef[:n]
        print("%s: %d frames" % (clip, n), flush=True)

        # GT + LR in RAM, EDVR computed once and reused for both variants
        gts = [cv2.imread(f) for f in gtf]
        H, W = gts[0].shape[:2]
        lrs = [down4(g) for g in gts]
        tls = [rd("/tmp/__none__") if False else None for _ in range(0)]
        lr_t = [torch.from_numpy(np.array(cv2.cvtColor(l, cv2.COLOR_BGR2RGB),
                                          dtype=np.float32) / 255.0).permute(2, 0, 1)
                for l in lrs]
        edvs = edvr_seq(lr_t, n)
        del lr_t

        j = min(40, n - 1)
        bic_j = cv2.resize(lrs[j], (W, H), interpolation=cv2.INTER_CUBIC)
        xy = busy_box(gts[j], bic_j)

        variants = [True] if only_inset else [False, True]
        for inset in variants:
            suffix = "demo2x2_inset" if inset else "demo2x2"
            out = os.path.join(outdir, "%s_%s.mp4" % (clip, suffix))
            cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (PW * 2, PH * 2),
                   "-r", str(FPS), "-i", "-", "-an",
                   "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", out]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for k in range(n):
                    bic = cv2.resize(lrs[k], (W, H), interpolation=cv2.INTER_CUBIC)
                    teo = cv2.imread(tef[k])
                    hdr = "clip %s   frame %03d/%03d   (id %d)" % (clip, k + 1, n, k)
                    p.stdin.write(grid(gts[k], bic, edvs[k], teo, hdr, xy, inset).tobytes())
                p.stdin.close()
                err = p.stderr.read().decode("utf-8", "replace")
                rc = p.wait()
            finally:
                if p.poll() is None:
                    p.kill()
            if rc != 0:
                print("ffmpeg failed %s rc=%d\n%s" % (out, rc, err[-500:]))
                if os.path.exists(out):
                    os.remove(out)
            else:
                print("  -> %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6), flush=True)
        del gts, lrs, edvs


if __name__ == "__main__":
    main()
