#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Part 2 -- EDVR-L x4 (REDS) applied to the WILD videos.

Identical to p2_edvr.py except the data roots come from argv. The 5-frame sliding
window, the centre_frame_idx=2 clamping at the clip edges, and the metric code are
unchanged, so these numbers are directly comparable to the REDS table.

usage: python p2_edvr_wild.py GT_ROOT LR_ROOT OUT
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, glob, json, types, time
import numpy as np
import torch
import torchvision
import cv2

_mod = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as _F
for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = _mod

from basicsr.archs.edvr_arch import EDVR
import lpips

torch.backends.cudnn.benchmark = True

B = "/data/disk1/shared/aiaa3201"
CK = B + "/checkpoints/EDVR_REDS_SR_L/EDVR_L_x4_SR_REDS_official-9f5f5039.pth"
DEV = "cuda"
FLOW_SCALE = 4
ALEX = "/tmp/torch_home/hub/checkpoints/alexnet-owt-7be5be79.pth"

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


def psnr_mean(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(10.0 * np.log10(255.0 ** 2 /
                 np.maximum((d * d).mean(axis=tuple(range(1, d.ndim))), 1e-12))))


_HAVE_LPIPS = os.path.exists(ALEX)
_lpips_net = None
if _HAVE_LPIPS:
    _lpips_net = lpips.LPIPS(net="alex").to(DEV).eval()
    for p in _lpips_net.parameters():
        p.requires_grad_(False)


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).to(DEV).float().div_(127.5).sub_(1.0).permute(0, 3, 1, 2)


def lpips_mean(a, b, bs=25):
    vals = []
    for i in range(0, len(a), bs):
        with torch.no_grad():
            d = _lpips_net(_t(a[i:i + bs]), _t(b[i:i + bs]))
        vals.extend(d.reshape(-1).cpu().numpy().tolist())
    return float(np.mean(vals))


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
    return cv2.remap(src, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def tlpips(frames, grays, gx, gy, stride=2):
    al, bl = [], []
    for i in range(1, len(frames), stride):
        al.append(warp_onto(frames[i - 1], grays[i], grays[i - 1], gx, gy))
        bl.append(frames[i])
    return lpips_mean(np.stack(al), np.stack(bl))


ck = torch.load(CK, map_location="cpu", weights_only=False)["params"]
model = EDVR(num_in_ch=3, num_out_ch=3, num_feat=128, num_frame=5,
             deformable_groups=8, num_extract_block=5, num_reconstruct_block=40,
             center_frame_idx=2, hr_in=False, with_predeblur=False, with_tsa=True)
model.load_state_dict(ck, strict=True)
model = model.cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)


def rd(p):
    a = np.array(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def edvr_clip(lr_dir, n):
    out = []
    for c in range(n):
        win = [max(0, min(n - 1, c + d - 2)) for d in range(5)]
        x = torch.stack([rd("%s/%08d.png" % (lr_dir, w)) for w in win]).unsqueeze(0).cuda()
        with torch.no_grad():
            y = model(x)
        y = y[0].clamp_(0, 1).mul_(255).round_().byte().permute(1, 2, 0).cpu().numpy()
        out.append(y)   # RGB
    return out


if __name__ == "__main__":
    GT_ROOT, LR_ROOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(OUT, exist_ok=True)
    clips = sorted(d for d in os.listdir(GT_ROOT) if os.path.isdir(os.path.join(GT_ROOT, d)))
    per = {}
    print("EDVR wild | clips=%d | lpips=%s" % (len(clips), _HAVE_LPIPS), flush=True)
    for ci, clip in enumerate(clips):
        t0 = time.time()
        gtfiles = sorted(glob.glob(os.path.join(GT_ROOT, clip, "*.png")))
        gt = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in gtfiles]
        n = len(gt)
        out = edvr_clip(os.path.join(LR_ROOT, clip), n)
        H, W = gt[0].shape[:2]
        assert out[0].shape[:2] == (H, W), "EDVR output %s != GT %s" % (out[0].shape[:2], (H, W))
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        gtY = np.stack([cv2.cvtColor(g, cv2.COLOR_RGB2YCrCb)[:, :, 0] for g in gt])
        pY = np.stack([cv2.cvtColor(p, cv2.COLOR_RGB2YCrCb)[:, :, 0] for p in out])
        gr = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in out]
        row = {"psnr": psnr_mean(pY, gtY), "ssim": float(ssim_gpu(pY, gtY).mean())}
        if _HAVE_LPIPS:
            row["lpips"] = lpips_mean(out, np.stack(gt))
            row["tlpips"] = tlpips(out, gr, gx, gy)
        per[clip] = row
        sd = os.path.join(OUT, "samples", clip)
        os.makedirs(sd, exist_ok=True)
        cv2.imwrite(os.path.join(sd, "edvr.png"), cv2.cvtColor(out[2], cv2.COLOR_RGB2BGR))
        print("[%d/%d] %s %4.0fs psnr %.2f ssim %.4f lpips %.4f tlpips %.4f"
              % (ci + 1, len(clips), clip, time.time() - t0, row["psnr"], row["ssim"],
                 row.get("lpips", -1), row.get("tlpips", -1)), flush=True)
        json.dump({"method": "edvr", "per_clip": per},
                  open(os.path.join(OUT, "metrics_edvr_wild.json"), "w"), indent=2)
    print("\n===== EDVR wild, %d clips =====" % len(clips), flush=True)
    av = {k: float(np.mean([r[k] for r in per.values()])) for k in per[clips[0]]}
    for k in ("psnr", "ssim", "lpips", "tlpips"):
        if k in av:
            print("%-8s %.3f" % (k, av[k]), flush=True)
    json.dump({"method": "edvr", "avg": av, "per_clip": per},
              open(os.path.join(OUT, "metrics_edvr_wild.json"), "w"), indent=2)
