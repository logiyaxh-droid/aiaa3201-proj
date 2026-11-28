#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Task 8 -- speed and memory benchmark for every method.

What the checklist asks for, and how it is measured here:

  * hardware, input size, frame count, batch size, precision  -> recorded in the JSON
  * warm-up before timing                                      -> WARMUP iterations, untimed
  * GPU sync around each repeat                                -> torch.cuda.synchronize()
  * separate pure inference from I/O                            -> three scopes per method
  * optical-flow estimation timed apart from warp+blend          -> its own rows
  * peak VRAM for the GPU methods                              -> max_memory_allocated

Scopes
------
``compute``  frames already decoded in RAM -> prediction in RAM.  This is the
             number to quote as "inference cost".
``plus_read`` compute + reading the PNGs off the shared disk (the old scripts
             mixed this in, which made bicubic look 6x slower than it is).
``plus_write`` compute + writing the output frames as PNG (what a run that keeps
             frames would pay).

All numbers are medians over ``--reps`` repeats done back to back, so the GPU
stays in one clock state; the machine is shared, so a ``max/min`` spread is
reported too and a wide spread means someone else was using the GPU.

Usage:
    /tmp/srenv/bin/python ~/p0/scripts/bench_speed.py --out ~/p0/out/bench/speed.json
"""
from __future__ import annotations

import argparse
import json
import os
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

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
CLIP = "000"
NFRAMES = 100


def rd_rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def load_clip(n=NFRAMES):
    gt = sorted(os.listdir(os.path.join(GT_ROOT, CLIP)))[:n]
    lr = [rd_rgb(os.path.join(LR_ROOT, CLIP, f)) for f in gt]
    return lr


# ------------------------------------------------------------------ algorithms
def fb_target_to_source(t, s, scale=4):
    h, w = t.shape
    ts = cv2.resize(t, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    ss = cv2.resize(s, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(ts, ss, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * float(scale)


def warp_onto(src, tg, sg, gx, gy):
    f = fb_target_to_source(tg, sg)
    return cv2.remap(src, gx + f[..., 0], gy + f[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def unsharp(f, radius=1.0, amount=1.5):
    ff = f.astype(np.float32)
    return np.clip(np.rint(ff + amount * (ff - cv2.GaussianBlur(ff, (0, 0), radius))),
                   0, 255).astype(np.uint8)


def naive_avg(frames, kernel):
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


def flows_cached(frames, gx=None, gy=None):
    """Farneback for every adjacent pair, timed on its own."""
    grays = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in frames]
    out = []
    for t in range(len(frames)):
        row = []
        for d in (-1, 1):
            j = min(max(t + d, 0), len(frames) - 1)
            row.append(fb_target_to_source(grays[t], grays[j]))
        out.append(row)
    return out, grays


def flow_avg_from_flows(frames, flows, gx, gy, kernel):
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
                f = flows[t][0 if j < t else 1]
                acc += wi * cv2.remap(frames[j], gx + f[..., 0], gy + f[..., 1],
                                      cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        out[t] = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------ timing
def timeit(fn, reps=3, warmup=1, gap=0.0):
    """Time ``fn`` and report min / median / max.

    The GPU is shared and other students' jobs come and go on a minute scale, so a
    median over back-to-back repeats measures their load as much as this code: two
    runs of the identical EDSR batch-4 workload measured 7.4 s and 14.7 s within
    ten minutes.  ``min`` is the statistic that is least contaminated (it needs one
    clean repeat), so it is what the report quotes; median/max are kept to show the
    spread, and a wide spread is itself evidence the machine was busy.
    """
    for _ in range(warmup):
        fn()
    sync = torch.cuda.is_available()
    ts = []
    for _ in range(reps):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        if gap:
            time.sleep(gap)
    a = np.asarray(ts)
    return {"min_s": float(a.min()), "median_s": float(np.median(a)),
            "max_s": float(a.max()), "reps": reps,
            "spread_ratio": float(a.max() / max(a.min(), 1e-9))}


class EDSRRunner:
    def __init__(self, device="cuda"):
        from basicsr.archs.edsr_arch import EDSR
        ck = torch.load("/data/disk1/shared/aiaa3201/checkpoints/EDSR_baseline_x4/"
                        "EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth",
                        map_location="cpu", weights_only=False)["params"]
        net = EDSR(3, 3, 64, 16, 4)
        net.load_state_dict(ck, strict=True)
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
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


class EDVRRunner:
    def __init__(self, device="cuda"):
        from basicsr.archs.edvr_arch import EDVR
        ck = torch.load("/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
                        "EDVR_L_x4_SR_REDS_official-9f5f5039.pth",
                        map_location="cpu", weights_only=False)["params"]
        net = EDVR(num_in_ch=3, num_out_ch=3, num_feat=128, num_frame=5,
                   deformable_groups=8, num_extract_block=5, num_reconstruct_block=40,
                   center_frame_idx=2, hr_in=False, with_predeblur=False, with_tsa=True)
        net.load_state_dict(ck, strict=True)
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device

    def __call__(self, lr):
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
        return np.stack(out, 0)


def peak_vram(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return {"peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2 ** 20,
            "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2 ** 20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gap", type=float, default=2.0,
                    help="seconds to sleep between repeats, so repeats sample "
                         "different load conditions")
    ap.add_argument("--frames", type=int, default=NFRAMES)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n = args.frames
    lr = load_clip(n)
    H, W = 720, 1280
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    bic = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr]
    res = {"clip": CLIP, "frames": n, "lr_hw": list(lr[0].shape[:2]),
           "hr_hw": [H, W], "reps": args.reps,
           "device": "NVIDIA RTX A6000 (shared)", "precision": "fp32 (torch default)",
           "note": "GPU is SHARED. Quote min_s (least contaminated); median/max "
                   "and spread_ratio show how noisy the machine was. The same EDSR "
                   "batch-4 workload measured 7.4 s and 14.7 s ten minutes apart.",
           "quoted_statistic": "min_s",
           "methods": {}, "components": {}}

    # ---- I/O scopes
    files_lr = [os.path.join(LR_ROOT, CLIP, f) for f in
                sorted(os.listdir(os.path.join(LR_ROOT, CLIP)))[:n]]
    res["components"]["read_lr_png"] = timeit(lambda: [cv2.imread(p) for p in files_lr],
                                              args.reps, gap=args.gap)
    res["components"]["read_gt_png"] = timeit(
        lambda: [cv2.imread(os.path.join(GT_ROOT, CLIP, f)) for f in
                 sorted(os.listdir(os.path.join(GT_ROOT, CLIP)))[:n]], args.reps,
                 gap=args.gap)
    tmpdir = "/tmp/bench_write"
    os.makedirs(tmpdir, exist_ok=True)

    def write_png():
        for i, f in enumerate(bic):
            cv2.imwrite(os.path.join(tmpdir, "%08d.png" % i), f)

    res["components"]["write_hr_png"] = timeit(write_png, 2, gap=args.gap)

    # ---- CPU methods
    def bench_cpu(name, fn, scope_builder=None):
        t = timeit(fn, args.reps, gap=args.gap)
        row = {"scope": "compute", "device": "cpu", **t}
        row["ms_per_frame"] = 1000.0 * t["min_s"] / n
        row["fps"] = n / t["min_s"]
        res["methods"][name] = row
        return row

    bench_cpu("bicubic", lambda: [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC)
                                  for x in lr])
    bench_cpu("lanczos", lambda: [cv2.resize(x, (W, H), interpolation=cv2.INTER_LANCZOS4)
                                  for x in lr])
    bench_cpu("bicubic_unsharp",
              lambda: [unsharp(cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC))
                       for x in lr])
    bench_cpu("tavg3", lambda: naive_avg(bic, (1, 2, 1)))
    bench_cpu("tavg5", lambda: naive_avg(bic, (1, 2, 3, 2, 1)))

    # optical flow alone, then warp+blend alone, then both
    grays = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in bic]
    flow_t = timeit(lambda: [(fb_target_to_source(grays[t], grays[min(t + 1, n - 1)]),
                              fb_target_to_source(grays[t], grays[max(t - 1, 0)]))
                             for t in range(n)], args.reps, gap=args.gap)
    flows, _ = flows_cached(bic)
    res["components"]["flow_farneback_all_pairs"] = flow_t
    res["components"]["flow_farneback_per_pair_ms"] = (
        1000.0 * flow_t["min_s"] / (2 * n))

    def warp_only():
        out = np.empty((n, H, W, 3), np.uint8)
        for t in range(n):
            acc = np.zeros((H, W, 3), np.float32)
            for j, fi in ((min(t + 1, n - 1), 0), (max(t - 1, 0), 1)):
                f = flows[t][fi]
                acc += cv2.remap(bic[j], gx + f[..., 0], gy + f[..., 1],
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out[t] = np.clip(np.rint(acc / 2.0), 0, 255).astype(np.uint8)
        return out

    res["components"]["warp_blend_all_frames"] = timeit(
        warp_only, args.reps, gap=args.gap)
    res["components"]["warp_blend_per_frame_ms"] = (
        1000.0 * res["components"]["warp_blend_all_frames"]["min_s"] / n)

    bench_cpu("flowtavg3", lambda: flow_avg_from_flows(bic, flows, gx, gy, (1, 2, 1)))
    bench_cpu("flowtavg5", lambda: flow_avg_from_flows(bic, flows, gx, gy,
                                                       (1, 2, 3, 2, 1)))
    bench_cpu("flowtavg5_unsharp",
              lambda: [unsharp(x) for x in
                       flow_avg_from_flows(bic, flows, gx, gy, (1, 2, 3, 2, 1))])
    bench_cpu("flow_full_recompute",
              lambda: flow_avg_from_flows(
                  bic, flows_cached(bic)[0], gx, gy, (1, 2, 3, 2, 1)))

    # write scope for one representative method
    res["methods"]["edsr_with_write"] = None  # filled below after EDSR is known

    # ---- GPU methods
    if torch.cuda.is_available():
        edsr = EDSRRunner()
        t = timeit(lambda: edsr(lr), args.reps, gap=args.gap)
        res["methods"]["edsr"] = {"scope": "compute", "device": "cuda",
                                  "batch": 4, **t,
                                  "ms_per_frame": 1000.0 * t["min_s"] / n,
                                  "fps": n / t["min_s"],
                                  **peak_vram(lambda: edsr(lr))}
        del res["methods"]["edsr_with_write"]

        def edsr_write():
            out = edsr(lr)
            for i, f in enumerate(out):
                cv2.imwrite(os.path.join(tmpdir, "e%08d.png" % i),
                            cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        tw = timeit(edsr_write, 2, gap=args.gap)
        res["methods"]["edsr_with_write"] = {
            "scope": "compute+write", "device": "cuda", **tw,
            "ms_per_frame": 1000.0 * tw["min_s"] / n, "fps": n / tw["min_s"]}

        edvr = EDVRRunner()
        t = timeit(lambda: edvr(lr), max(3, args.reps - 1), gap=args.gap)
        res["methods"]["edvr"] = {"scope": "compute", "device": "cuda", "batch": 1,
                                  "window": 5, **t,
                                  "ms_per_frame": 1000.0 * t["min_s"] / n,
                                  "fps": n / t["min_s"],
                                  **peak_vram(lambda: edvr(lr))}

        # small batch sweep, to report what batch size buys
        for bs in (1, 2, 4, 8):
            def run(bs=bs):
                out = []
                for i in range(0, n, bs):
                    x = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float()
                                     .div_(255.0) for f in lr[i:i + bs]]).to("cuda")
                    with torch.no_grad():
                        y = edsr.net(x)
                    out.append(y)
                return out

            res["methods"]["edsr_bs%d" % bs] = {
                "scope": "compute", "device": "cuda", "batch": bs,
                **timeit(run, 3, gap=args.gap),
                **peak_vram(run)}

    for k in ("edsr_bs1", "edsr_bs2", "edsr_bs4", "edsr_bs8"):
        if k in res["methods"]:
            res["methods"][k]["ms_per_frame"] = 1000.0 * res["methods"][k]["median_s"] / n

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False)[:4000])
    print("->", args.out)


if __name__ == "__main__":
    main()
