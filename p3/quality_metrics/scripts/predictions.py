# -*- coding: utf-8 -*-
"""Prediction providers for the P3 quality-metrics runs.

`full_eval.py` builds every Part-1 method and then throws the frames away, so the
new metrics have to re-generate them.  This module is that generator, lifted out
of `full_eval.py` unchanged (same kernels, same uint8 rounding, same flows) but
exposing one clip of frames at a time so DISTS / MUSIQ / DOVER can consume them
before they are released.

Nothing here is allowed to alter a pixel: every method's output must stay
bit-identical to what produced the stored P0 numbers.  ''old_metrics'' is the
regression gate that proves it.

Caches that already exist on disk are used as-is and preferred:
    bicubic        p0/out/pred/bicubic/<clip>            (only 000, 027)
    tecogan        /tmp/p2/tecogan/<clip>                 (all 30)
    edvr           /tmp/p2/edvr_frame/<clip> + p3/ev/edvr_frame/<clip>
"""
from __future__ import annotations

import glob
import os
import time

import cv2
import numpy as np

import common as C

PART1 = ["bicubic", "lanczos", "bicubic_unsharp", "edsr", "tavg3", "tavg5",
         "tavg5_unsharp", "flowtavg3", "flowtavg5", "flowtavg5_unsharp"]
PART2 = ["edvr", "tecogan"]
METHODS = PART1 + PART2

# existing on-disk prediction caches, in priority order
CACHE_DIRS = {
    "bicubic": [C.BICUBIC_PRED, os.path.join(C.ROOT, "cache", "pred", "bicubic")],
    "tecogan": [C.TECO_FRAMES, os.path.join(C.ROOT, "cache", "pred", "tecogan")],
    "edvr": [os.path.join("/data/disk1/home/ghe270/p3/ev", "edvr_frame"),
             C.EDVR_FRAMES, os.path.join(C.ROOT, "cache", "pred", "edvr")],
}


def cached_clip_dir(method: str, clip: str):
    for root in CACHE_DIRS.get(method, []):
        d = os.path.join(root, clip)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.png")):
            return d
    return None


# ---------------------------------------------------------------- Part-1 ops

def fb_target_to_source(target_gray, source_gray, scale=4):
    h, w = target_gray.shape
    ts = cv2.resize(target_gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    ss = cv2.resize(source_gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(ts, ss, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * float(scale)


def warp_onto(src, target_gray, source_gray, gx, gy):
    f = fb_target_to_source(target_gray, source_gray)
    return cv2.remap(src, gx + f[..., 0], gy + f[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def naive_avg(frames, kernel):
    wt = np.asarray(kernel, np.float32)
    wt /= wt.sum()
    k = len(wt) // 2
    n = len(frames)
    h, w = frames[0].shape[:2]
    out = np.empty((n, h, w, 3), np.uint8)
    for t in range(n):
        acc = np.zeros((h, w, 3), np.float32)
        for d, w_i in enumerate(wt):
            j = min(max(t + d - k, 0), n - 1)
            acc += w_i * frames[j].astype(np.float32)
        out[t] = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
    return out


def flow_avg_blockwise(frames, grays, gx, gy, kernel, log=None):
    wt = np.asarray(kernel, np.float32)
    wt /= wt.sum()
    k = len(wt) // 2
    n = len(frames)
    h, w = frames[0].shape[:2]
    out = np.empty((n, h, w, 3), np.uint8)
    for t in range(n):
        acc = np.zeros((h, w, 3), np.float32)
        for d, w_i in enumerate(wt):
            j = min(max(t + d - k, 0), n - 1)
            acc += w_i * (frames[t].astype(np.float32) if j == t
                          else warp_onto(frames[j], grays[t], grays[j], gx, gy)
                          .astype(np.float32))
        out[t] = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
        if log and (t + 1) % 25 == 0:
            log("      flow-avg %d/%d" % (t + 1, n))
    return out


def unsharp(f, radius=1.0, amount=1.5):
    ff = f.astype(np.float32)
    return np.clip(np.rint(ff + amount * (ff - cv2.GaussianBlur(ff, (0, 0), radius))),
                   0, 255).astype(np.uint8)


# ---------------------------------------------------------------- runners

class EDSRRunner:
    def __init__(self, device="cuda"):
        from basicsr.archs.edsr_arch import EDSR
        ck = torch_load(C.CK_EDSR)["params"]
        self.net = EDSR(3, 3, 64, 16, 4)
        self.net.load_state_dict(ck, strict=True)
        self.net = self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device

    def __call__(self, lr, bs=4):
        import torch
        out = []
        for i in range(0, len(lr), bs):
            x = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float().div_(255.0)
                             for f in lr[i:i + bs]]).to(self.device)
            with torch.no_grad():
                y = self.net(x)
            y = y.clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).cpu().numpy()
            out.extend(y[k] for k in range(y.shape[0]))
        return out


class EDVRRunner:
    """EDVR-L x4 (REDS), 5-frame sliding window, centre frame idx 2."""

    def __init__(self, device="cuda"):
        from basicsr.archs.edvr_arch import EDVR
        ck = torch_load(C.CK_EDVR)["params"]
        self.net = EDVR(num_in_ch=3, num_out_ch=3, num_feat=128, num_frame=5,
                        deformable_groups=8, num_extract_block=5,
                        num_reconstruct_block=40, center_frame_idx=2,
                        hr_in=False, with_predeblur=False, with_tsa=True)
        self.net.load_state_dict(ck, strict=True)
        self.net = self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device

    def clip(self, lr):
        import torch
        n = len(lr)
        out = []
        for c in range(n):
            win = [max(0, min(n - 1, c + d - 2)) for d in range(5)]
            x = torch.stack([torch.from_numpy(lr[w]).permute(2, 0, 1).float().div_(255.0)
                             for w in win]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                y = self.net(x)
            out.append(y[0].clamp_(0, 1).mul_(255).round_().byte()
                       .permute(1, 2, 0).cpu().numpy())
        return out


def torch_load(path, tries=4):
    import torch
    for a in range(tries):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:                       # shared disk short reads
            print("ckpt retry %d/%d: %s" % (a + 1, tries, exc), flush=True)
            time.sleep(3)
    raise RuntimeError("cannot read checkpoint %s" % path)


# ---------------------------------------------------------------- per clip

class ClipFrames:
    """One clip's GT + LR, plus a generator per method, bounded to one method.

    Peak memory is one clip's worth of 720p frames (~276 MB) rather than
    everything at once, which is what keeps this viable on a shared machine.
    """

    def __init__(self, clip: str, device: str = "cuda", log=print):
        self.clip = clip
        self.device = device
        self.log = log
        gt_dir = os.path.join(C.GT_ROOT, clip)
        self.ids = C.clip_frame_ids(gt_dir)
        self.gt = C.read_clip(gt_dir, self.ids)
        n = len(self.gt)
        self.H, self.W = self.gt[0].shape[:2]
        lr_dir = os.path.join(C.LR_ROOT, clip)
        self.lr = C.read_clip(lr_dir, self.ids)
        self.gx, self.gy = np.meshgrid(np.arange(self.W, dtype=np.float32),
                                       np.arange(self.H, dtype=np.float32))
        self._bic = None
        self._grays = None
        self._product = {}
        self._runners = {}

    def need_bic(self):
        if self._bic is None:
            self._bic = [cv2.resize(x, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
                         for x in self.lr]
            self._grays = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in self._bic]
        return self._bic

    def _runner(self, key):
        if key not in self._runners:
            self._runners[key] = (EDSRRunner(self.device) if key == "edsr"
                                  else EDVRRunner(self.device))
        return self._runners[key]

    def _mk_product(self, name):
        if name in self._product:
            return self._product[name]
        bic = self.need_bic()
        if name == "tavg3":
            out = naive_avg(bic, (1, 2, 1))
        elif name == "tavg5":
            out = naive_avg(bic, (1, 2, 3, 2, 1))
        elif name == "flowtavg3":
            out = flow_avg_blockwise(bic, self._grays, self.gx, self.gy, (1, 2, 1),
                                     self.log)
        elif name == "flowtavg5":
            out = flow_avg_blockwise(bic, self._grays, self.gx, self.gy,
                                     (1, 2, 3, 2, 1), self.log)
        else:
            raise KeyError(name)
        self._product[name] = out
        return out

    def frames(self, method: str):
        """-> (pred_list, source) for one method; source says cache vs computed."""
        cache = cached_clip_dir(method, self.clip)
        if cache:
            return C.read_clip(cache, self.ids), "cache:" + cache
        if method == "bicubic":
            return self.need_bic(), "computed"
        if method == "lanczos":
            return [cv2.resize(x, (self.W, self.H), interpolation=cv2.INTER_LANCZOS4)
                    for x in self.lr], "computed"
        if method == "bicubic_unsharp":
            return [unsharp(x) for x in self.need_bic()], "computed"
        if method == "edsr":
            return self._runner("edsr")(self.lr), "computed"
        if method == "tavg3":
            return self._mk_product("tavg3"), "computed"
        if method == "tavg5":
            return self._mk_product("tavg5"), "computed"
        if method == "tavg5_unsharp":
            return [unsharp(x) for x in self._mk_product("tavg5")], "computed"
        if method == "flowtavg3":
            return self._mk_product("flowtavg3"), "computed"
        if method == "flowtavg5":
            return self._mk_product("flowtavg5"), "computed"
        if method == "flowtavg5_unsharp":
            return [unsharp(x) for x in self._mk_product("flowtavg5")], "computed"
        if method == "edvr":
            return self._runner("edvr").clip(self.lr), "computed"
        if method == "tecogan":
            raise FileNotFoundError("clip %s: no TecoGAN cache" % self.clip)
        raise ValueError("unknown method %r" % method)

    def release(self):
        self._product.clear()
        self._bic = None
        self._grays = None
        self._runners.clear()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
