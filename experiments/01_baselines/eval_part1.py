#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Part 1 -- corrected temporal metrics + flow-aligned fusion.

WARPING CONVENTION, calibrated against a synthetic known-shift test:
  To align source frame S onto target frame T (output(x) = S(x + f)):
      f = calcOpticalFlowFarneback(gray_T, gray_S)      # target -> source flow
      warped = remap(S, gx + f[0], gy + f[1])
  `remap(S, flow(S,T))` is WRONG -- it raises the error by ~30% instead of
  lowering it. Verified with a known 12 px shift: correct L1 1.36 vs 34.85.

Flow is estimated at 1/4 resolution: 86.6% L1 reduction vs 70.3% at full
resolution, and 58x cheaper (17 ms vs 1000 ms per pair).

Regenerates (a) tLPIPS for every Part-1 method and (b) flow-aligned averaging.
PSNR/SSIM/LPIPS are flow-independent and come from metrics_part1.json.
"""
import os, sys, json, types, glob, time
import numpy as np
import torch
import cv2

import torchvision
_mod = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as _F
for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = _mod

from basicsr.archs.edsr_arch import EDSR
import lpips

torch.backends.cudnn.benchmark = True

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
CK = ("/data/disk1/shared/aiaa3201/checkpoints/EDSR_baseline_x4/"
      "EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth")
OUT = "/tmp/p1_part1"
DEV = "cuda"
FLOW_SCALE = 4

_PAD = 3
_K = torch.ones(1, 1, 7, 7, dtype=torch.float64, device=DEV) / 49.0
_COV = 49.0 / 48.0


def ssim_gpu(a_u8, b_u8, data_range=255.0):
    a = torch.from_numpy(np.ascontiguousarray(a_u8)).to(DEV).double().unsqueeze(1)
    b = torch.from_numpy(np.ascontiguousarray(b_u8)).to(DEV).double().unsqueeze(1)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    f = lambda x: torch.nn.functional.conv2d(x, _K)
    ux, uy = f(a), f(b)
    uxx = _COV * (f(a * a) - ux * ux)
    uyy = _COV * (f(b * b) - uy * uy)
    uxy = _COV * (f(a * b) - ux * uy)
    m = ((2 * ux * uy + C1) * (2 * uxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (uxx + uyy + C2))
    return m.mean(dim=(1, 2, 3)).cpu().numpy()


_net = lpips.LPIPS(net="alex").to(DEV).eval()
for p in _net.parameters():
    p.requires_grad_(False)


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).to(DEV).float().div_(127.5).sub_(1.0).permute(0, 3, 1, 2)


def lpips_mean(a, b, bs=25):
    vals = []
    for i in range(0, len(a), bs):
        with torch.no_grad():
            d = _net(_t(a[i:i + bs]), _t(b[i:i + bs]))
        vals.extend(d.reshape(-1).cpu().numpy().tolist())
    return float(np.mean(vals))


def psnr_mean(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(10.0 * np.log10(255.0 ** 2 / np.maximum((d * d).mean(axis=tuple(range(1, d.ndim))), 1e-12))))


def fb_target_to_source(target_gray, source_gray, scale=FLOW_SCALE):
    h, w = target_gray.shape
    if scale == 1:
        return cv2.calcOpticalFlowFarneback(target_gray, source_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    ts = cv2.resize(target_gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    ss = cv2.resize(source_gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(ts, ss, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * float(scale)


def warp_onto(src, target_gray, src_gray, gx, gy):
    f = fb_target_to_source(target_gray, src_gray)
    return cv2.remap(src, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def tlpips(frames, grays, gx, gy, stride=2):
    al, bl = [], []
    for i in range(1, len(frames), stride):
        al.append(warp_onto(frames[i - 1], grays[i], grays[i - 1], gx, gy))
        bl.append(frames[i])
    return lpips_mean(np.stack(al), np.stack(bl))


def flow_temporal_avg(frames, grays, gx, gy, kernel=(1, 2, 3, 2, 1)):
    h, w = frames[0].shape[:2]
    wt = np.asarray(kernel, np.float32)
    wt /= wt.sum()
    k = len(wt) // 2
    n = len(frames)
    cache = {}

    def get(t, j):
        if j == t:
            return frames[t].astype(np.float32)
        if (t, j) not in cache:
            cache[(t, j)] = warp_onto(frames[j], grays[t], grays[j], gx, gy).astype(np.float32)
        return cache[(t, j)]

    out = np.zeros((n, h, w, 3), np.float32)
    for t in range(n):
        acc = np.zeros((h, w, 3), np.float32)
        for d, w_i in enumerate(wt):
            acc += w_i * get(t, min(max(t + d - k, 0), n - 1))
        out[t] = acc
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def naive_avg(frames, kernel=(1, 2, 3, 2, 1)):
    wt = np.asarray(kernel, np.float32)
    wt /= wt.sum()
    k = len(wt) // 2
    n = len(frames)
    f = np.stack([x.astype(np.float32) for x in frames])
    pad = np.concatenate([f[:1]] * k + [f] + [f[-1:]] * k, 0)
    return np.clip(np.rint(sum(wt[i] * pad[i:i + n] for i in range(len(wt)))), 0, 255).astype(np.uint8)


def unsharp(f, radius=1.0, amount=1.5):
    ff = f.astype(np.float32)
    return np.clip(np.rint(ff + amount * (ff - cv2.GaussianBlur(ff, (0, 0), radius))), 0, 255).astype(np.uint8)


_ck = torch.load(CK, map_location="cpu", weights_only=False)["params"]
edsr = EDSR(3, 3, 64, 16, 4)
edsr.load_state_dict(_ck, strict=True)
edsr = edsr.to(DEV).eval()
for p in edsr.parameters():
    p.requires_grad_(False)


def edsr_infer(lr_list, bs=4):
    out = []
    for i in range(0, len(lr_list), bs):
        x = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float().div_(255.0)
                         for f in lr_list[i:i + bs]]).to(DEV)
        with torch.no_grad():
            y = edsr(x)
        y = y.clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).cpu().numpy()
        out.extend(y[k] for k in range(y.shape[0]))
    return out


METHODS = ["bicubic", "lanczos", "bicubic_unsharp", "edsr", "tavg3", "tavg5", "tavg5_unsharp",
           "flowtavg3", "flowtavg5", "flowtavg5_unsharp"]

if __name__ == "__main__":
    clips = sorted(d for d in os.listdir(GT_ROOT) if os.path.isdir(os.path.join(GT_ROOT, d)))
    LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else len(clips)
    START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    clips = clips[START:START + LIMIT]

    acc = {m: {k: [] for k in ("psnr", "ssim", "lpips", "tlpips")} for m in METHODS}
    per = {}
    t_all = time.time()
    print("clips: %d  flow_scale=%d" % (len(clips), FLOW_SCALE), flush=True)

    for ci, clip in enumerate(clips):
        t0 = time.time()
        files = sorted(glob.glob(os.path.join(GT_ROOT, clip, "*.png")))
        gt = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
        lr = [cv2.cvtColor(cv2.imread(os.path.join(LR_ROOT, clip, os.path.basename(f))), cv2.COLOR_BGR2RGB) for f in files]
        H, W = gt[0].shape[:2]
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

        bic = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr]
        gbic = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in bic]
        f3 = flow_temporal_avg(bic, gbic, gx, gy, (1, 2, 1))
        f5 = flow_temporal_avg(bic, gbic, gx, gy, (1, 2, 3, 2, 1))

        P = {
            "bicubic": bic,
            "lanczos": [cv2.resize(x, (W, H), interpolation=cv2.INTER_LANCZOS4) for x in lr],
            "edsr": edsr_infer(lr),
            "tavg3": naive_avg(bic, (1, 2, 1)),
            "tavg5": naive_avg(bic, (1, 2, 3, 2, 1)),
            "flowtavg3": f3,
            "flowtavg5": f5,
        }
        P["bicubic_unsharp"] = [unsharp(x) for x in bic]
        P["tavg5_unsharp"] = [unsharp(x) for x in P["tavg5"]]
        P["flowtavg5_unsharp"] = np.stack([unsharp(x) for x in f5])

        gt_arr = np.stack(gt)
        gtY = np.stack([cv2.cvtColor(g, cv2.COLOR_RGB2YCrCb)[:, :, 0] for g in gt])

        row = {}
        for m in METHODS:
            arr = P[m]
            gr = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in arr]
            pY = np.stack([cv2.cvtColor(p, cv2.COLOR_RGB2YCrCb)[:, :, 0] for p in arr])
            row[m] = {"psnr": psnr_mean(pY, gtY),
                      "ssim": float(ssim_gpu(pY, gtY).mean()),
                      "lpips": lpips_mean(arr, gt_arr),
                      "tlpips": tlpips(arr, gr, gx, gy)}
            for k in row[m]:
                acc[m][k].append(row[m][k])

        if clip in ("000", "027"):
            for m in METHODS:
                cv2.imwrite("%s/samples/%s_%s.png" % (OUT, clip, m), cv2.cvtColor(P[m][2], cv2.COLOR_RGB2BGR))

        per[clip] = row
        print("[%2d/%d] %s %4.0fs " % (ci + 1, len(clips), clip, time.time() - t0) +
              " ".join("%s:%.2f/%.3f" % (m, row[m]["psnr"], row[m]["tlpips"])
                       for m in ("bicubic", "edsr", "tavg5", "flowtavg5")), flush=True)
        json.dump({"per_clip": per, "avg": {m: {k: float(np.mean(acc[m][k])) for k in acc[m]} for m in METHODS}},
                  open(OUT + "/metrics_temporal_partial.json", "w"), indent=2)

    print("\n===== REDS val, %d clips | flow direction calibrated, scale=%d =====" % (len(clips), FLOW_SCALE), flush=True)
    print("%-19s %8s %8s %8s %8s" % ("method", "PSNR", "SSIM", "LPIPS", "tLPIPS"), flush=True)
    summ = {}
    for m in METHODS:
        s = {k: float(np.mean(acc[m][k])) for k in acc[m]}
        summ[m] = s
        print("%-19s %8.3f %8.5f %8.4f %8.4f" % (m, s["psnr"], s["ssim"], s["lpips"], s["tlpips"]), flush=True)
    print("total %.1f min | %.0f s/clip" % ((time.time() - t_all) / 60, (time.time() - t_all) / len(clips)), flush=True)
    json.dump({"n_clips": len(clips), "flow_scale": FLOW_SCALE, "avg": summ, "per_clip": per},
              open(OUT + "/metrics_part1_final.json", "w"), indent=2)
    print("saved " + OUT + "/metrics_part1_final.json", flush=True)
