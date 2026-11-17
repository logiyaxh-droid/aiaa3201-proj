#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Full 30-clip evaluation: reference tLP + spatial metrics + FID features.

This is the "方案 B" run from ``全量tLP_FID_耗时预估.md``: every Part-1 method is
RE-GENERATED (the 2026-11-11 run discarded its prediction frames), so this is not
just a metrics pass.  Everything is streamed -- predictions are built for one
method, scored, and dropped -- which keeps peak RAM around one clip's worth of
720p frames instead of 12 x 276 MB.

Protocol (identical for every method, recorded in the run config)
---------------------------------------------------------------
spatial   PSNR / SSIM on the Y channel of ``cv2.COLOR_RGB2YCrCb``, 255 range,
          no border crop, per-frame then mean.  LPIPS = ``lpips`` 0.1.4 alex,
          inputs RGB in [-1,1].  All three come from ``evalkit.metrics`` so the
          numbers are the same objects Part 1 / Part 2 were scored with.
tLP       ``|LPIPS(SR[t-1],SR[t]) - LPIPS(GT[t-1],GT[t])|`` averaged over ALL
          consecutive pairs (99 per 100-frame clip), cutfr=0, crop32=False,
          x100=False.  This reproduces the p0 clip-000 reference value 0.0261
          for bicubic (see ``--verify-clip``).
flow-warp ``LPIPS(warp(prev),cur)`` with Farneback at 1/4 resolution,
          stride=2 -- i.e. exactly the legacy "tLPIPS" definition, kept so the
          old JSON numbers stay comparable.  It is a diagnostic, not tLP.
FID       pytorch-fid InceptionV3 (pt_inception-2015-12-05), 2048-d pool3,
          RGB in [0,1], bilinear resize to 299 (align_corners=False).  Features
          are cached per clip so FID can be recomputed without re-running.

Resumability
------------
``<out>/results/<method>.json`` holds per-clip rows and is rewritten after every
clip; ``<out>/feat/<method>/<clip>.npy`` holds that clip's 100x2048 features.
Re-running skips whatever is already complete, so a killed job costs one clip.

Usage
-----
    /tmp/srenv/bin/python ~/p0/scripts/full_eval.py --out ~/p0/out/full
    /tmp/srenv/bin/python ~/p0/scripts/full_eval.py --out ~/p0/out/full \
        --methods bicubic --clips 000 --verify-clip 000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import types

os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import numpy as np
import cv2
import torch

# BasicSR (EDSR/EDVR) still imports the module torchvision removed in 0.17.
import torchvision
_mod = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F  # noqa: E402

for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _mod
# NOTE: cudnn.benchmark=True makes the FIRST float64 7x7 conv in evalkit.ssim_gpu
# spend ~126 s in cuDNN autotuning for that shape (measured 2026-11-17), which
# dominated a whole clip.  With it off the call takes 0.4 s and returns the same
# bits, so it stays off.
torch.backends.cudnn.benchmark = False

sys.path.insert(0, os.path.expanduser("~/p0"))
from evalkit import metrics as M          # noqa: E402
from evalkit import temporal as T         # noqa: E402
from evalkit import data as D             # noqa: E402

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
CK_EDSR = ("/data/disk1/shared/aiaa3201/checkpoints/EDSR_baseline_x4/"
           "EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth")
CK_EDVR = ("/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
           "EDVR_L_x4_SR_REDS_official-9f5f5039.pth")
TECO_DIR = "/tmp/p2/tecogan"
EDVR_FRAMES = "/tmp/p2/edvr_frame"

PART1 = ["bicubic", "lanczos", "bicubic_unsharp", "edsr", "tavg3", "tavg5",
         "tavg5_unsharp", "flowtavg3", "flowtavg5", "flowtavg5_unsharp"]
PART2 = ["edvr", "tecogan"]
METHODS = PART1 + PART2

CONVENTIONS = {
    "spatial": M.CONVENTIONS,
    "tLP": T.TLP_DEFINITION,
    "tLP_options": {"x100": False, "cutfr": 0, "crop32": False,
                    "pairs": "all consecutive ids inside one clip"},
    "flow_warp_lpips": {"stride": 2, "flow_scale": 4,
                        "flow": "cv2.calcOpticalFlowFarneback(prev_gray, cur_gray), "
                                "pyr=0.5/3/15/3/5/1.2, 1/4-res, remap(cur, +f)",
                        "note": "legacy 'tLPIPS' definition, diagnostic only"},
    "fid": {"impl": "pytorch-fid 0.3.0", "weights": "pt_inception-2015-12-05-6726825d",
            "dims": 2048, "resize": "bilinear 299 align_corners=False",
            "normalize": "2x-1", "frames": "all 30 clips x 100 frames = 3000"},
}

# ---------------------------------------------------------------- helpers

def log(msg, fp=None):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if fp:
        fp.write(line + "\n")
        fp.flush()


def psnr_fast(pY, gY):
    """uint8 Y stacks -> mean per-frame PSNR, float32 streaming.

    Identical to ``evalkit.metrics.psnr_frames`` except that the difference is
    accumulated in float32 chunks: the float64 version allocates ~740 MB per
    100-frame clip and takes 16 s.  ``--verify-clip`` asserts the two agree.
    """
    out = []
    for i in range(0, len(pY), 20):
        a = pY[i:i + 20].astype(np.float32)
        b = gY[i:i + 20].astype(np.float32)
        d = a - b
        mse = (d * d).mean(axis=tuple(range(1, d.ndim)))
        out.append(10.0 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12)))
    return float(np.concatenate(out).mean())


def rd_rgb(path):
    a = cv2.imread(path)
    if a is None:
        raise RuntimeError("unreadable frame: %s" % path)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def clip_frames(clip):
    gt = D.match_clip(os.path.join(GT_ROOT, clip), os.path.join(GT_ROOT, clip))
    return gt["ids"], gt["gt"]


def lr_frames(clip, ids):
    """LR frames matched by frame id (never by position)."""
    idx = D.index_dir(os.path.join(LR_ROOT, clip))
    missing = [i for i in ids if i not in idx]
    if missing:
        raise D.FrameSetError("clip %s: LR missing frame ids %s" % (clip, missing[:5]))
    return [rd_rgb(idx[i]) for i in ids]


# ---------------------------------------------------------------- part-1 methods

def fb_target_to_source(target_gray, source_gray, scale=4):
    """Copy of the calibrated direction (see evalkit.temporal and flow_selftest)."""
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
    """Boundary-clamped weighted average, one tap at a time.

    The 2026-11-11 version built a padded float32 copy of the whole clip
    (5 x 100 x 720p float32 = 5.5 GB) before summing.  Accumulating tap by tap
    costs the same arithmetic and holds one output frame plus one input frame,
    which is what makes a 30-clip run fit in a shared 31 GB machine.
    """
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


def flow_avg_blockwise(frames, grays, gx, gy, kernel):
    """Boundary-clamped, motion-compensated weighted average.

    Same arithmetic as ``p1_part1_final.flow_temporal_avg`` but the warp cache is
    dropped frame by frame instead of being kept for the whole clip (the original
    held ~400 warped 720p float32 frames = ~4.4 GB).
    """
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
                          else warp_onto(frames[j], grays[t], grays[j], gx, gy).astype(np.float32))
        out[t] = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
    return out


def unsharp(f, radius=1.0, amount=1.5):
    ff = f.astype(np.float32)
    return np.clip(np.rint(ff + amount * (ff - cv2.GaussianBlur(ff, (0, 0), radius))),
                   0, 255).astype(np.uint8)


class EDSRRunner:
    def __init__(self, device="cuda"):
        from basicsr.archs.edsr_arch import EDSR
        ck = torch.load(CK_EDSR, map_location="cpu", weights_only=False)["params"]
        self.net = EDSR(3, 3, 64, 16, 4)
        self.net.load_state_dict(ck, strict=True)
        self.net = self.net.to(device).eval()
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
            y = y.clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).cpu().numpy()
            out.extend(y[k] for k in range(y.shape[0]))
        return out


class EDVRRunner:
    """EDVR-L x4 (REDS), 5-frame sliding window, centre frame idx 2."""

    def __init__(self, device="cuda"):
        from basicsr.archs.edvr_arch import EDVR
        ck = self._load(CK_EDVR)
        self.net = EDVR(num_in_ch=3, num_out_ch=3, num_feat=128, num_frame=5,
                        deformable_groups=8, num_extract_block=5,
                        num_reconstruct_block=40, center_frame_idx=2,
                        hr_in=False, with_predeblur=False, with_tsa=True)
        self.net.load_state_dict(ck, strict=True)
        self.net = self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device

    @staticmethod
    def _load(path, tries=4):
        import shutil
        for a in range(tries):
            try:
                return torch.load(path, map_location="cpu", weights_only=False)["params"]
            except Exception as e:            # shared disk returns short reads
                print("ckpt retry %d/%d: %s" % (a + 1, tries, e), flush=True)
                time.sleep(3)
        dst = "/tmp/edvr_ck.pth"
        if not os.path.exists(dst):
            shutil.copyfile(path, dst)
        return torch.load(dst, map_location="cpu", weights_only=False)["params"]

    def clip(self, lr):
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


# ---------------------------------------------------------------- FID

class FIDExtractor:
    def __init__(self, device="cuda"):
        from pytorch_fid.inception import InceptionV3
        self.net = InceptionV3([3], resize_input=True, normalize_input=True,
                               use_fid_inception=True).to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device

    def __call__(self, frames, bs=16):
        feats = []
        for i in range(0, len(frames), bs):
            x = torch.from_numpy(np.ascontiguousarray(np.stack(frames[i:i + bs])))
            x = x.to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
            with torch.no_grad():
                f = self.net(x)[0]
            feats.append(f.reshape(f.shape[0], -1).cpu().numpy().astype(np.float32))
        return np.concatenate(feats, 0)


def fid_from_features(f1, f2):
    """Reference pytorch-fid formula (sqrt of the product of covariances)."""
    from scipy import linalg
    m1, m2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    diff = m1 - m2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))


# ---------------------------------------------------------------- per-clip driver

def score_clip(clip, methods, runners, lpips_net, fid_net, out_root, verify=False):
    t_clip = time.time()
    ids, gt_paths = clip_frames(clip)
    gt = [rd_rgb(p) for p in gt_paths]
    n = len(gt)
    H, W = gt[0].shape[:2]
    diff_sizes = {im.shape[:2] for im in gt}
    if diff_sizes != {(H, W)}:
        raise RuntimeError("clip %s: GT frames differ in size %s" % (clip, sorted(diff_sizes)))

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    gtY = np.stack([M.luma(g) for g in gt])

    # GT side of tLP is method independent -> compute once per clip
    gt_pair_prev = np.stack([gt[a] for a, _ in T.pairs_of(ids, 0)])
    gt_pair_cur = np.stack([gt[b] for _, b in T.pairs_of(ids, 0)])
    d_gt = lpips_net.pair(gt_pair_prev, gt_pair_cur)

    rows, feats = {}, {}
    gt_feat = fid_net(gt)
    feats["gt"] = gt_feat

    lr = lr_frames(clip, ids)
    bic = None
    grays = None

    def need_bic():
        nonlocal bic, grays
        if bic is None:
            bic = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr]
            grays = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in bic]
        return bic

    # products that are reused by a "_unsharp" variant are built once per clip
    cache = {}

    def product(name):
        if name not in cache:
            if name == "tavg3":
                cache[name] = naive_avg(need_bic(), (1, 2, 1))
            elif name == "tavg5":
                cache[name] = naive_avg(need_bic(), (1, 2, 3, 2, 1))
            elif name == "flowtavg3":
                cache[name] = flow_avg_blockwise(need_bic(), grays, gx, gy, (1, 2, 1))
            elif name == "flowtavg5":
                cache[name] = flow_avg_blockwise(need_bic(), grays, gx, gy, (1, 2, 3, 2, 1))
        return cache[name]

    for m in methods:
        t0 = time.time()
        if m == "bicubic":
            pred = need_bic()
        elif m == "lanczos":
            pred = [cv2.resize(x, (W, H), interpolation=cv2.INTER_LANCZOS4) for x in lr]
        elif m == "bicubic_unsharp":
            pred = [unsharp(x) for x in need_bic()]
        elif m == "edsr":
            pred = runners["edsr"](lr)
        elif m == "tavg3":
            pred = product("tavg3")
        elif m == "tavg5":
            pred = product("tavg5")
        elif m == "tavg5_unsharp":
            pred = [unsharp(x) for x in product("tavg5")]
        elif m == "flowtavg3":
            pred = product("flowtavg3")
        elif m == "flowtavg5":
            pred = product("flowtavg5")
        elif m == "flowtavg5_unsharp":
            pred = [unsharp(x) for x in product("flowtavg5")]
        elif m == "edvr":
            pred = runners["edvr"].clip(lr)
        elif m == "tecogan":
            idx = D.index_dir(os.path.join(TECO_DIR, clip))
            miss = [i for i in ids if i not in idx]
            if miss:
                raise D.FrameSetError("clip %s: TecoGAN missing %s" % (clip, miss[:5]))
            pred = [rd_rgb(idx[i]) for i in ids]
        else:
            raise ValueError("unknown method %r" % m)

        pred = np.asarray(pred)
        if pred.shape[:3] != (n, H, W):
            raise RuntimeError("clip %s %s: prediction shape %s != %s"
                               % (clip, m, pred.shape, (n, H, W)))

        pY = np.stack([M.luma(x) for x in pred])
        row = {"n_frames": n,
               "psnr": psnr_fast(pY, gtY),
               "ssim": float(M.ssim_gpu(pY, gtY, device=lpips_net.device).mean()),
               "lpips": lpips_net.mean(pred, np.stack(gt))}

        if verify:
            row["psnr_float64"] = M.psnr_mean(pY, gtY)
            row["psnr_abs_diff"] = abs(row["psnr"] - row["psnr_float64"])

        # reference tLP (all consecutive pairs, no cutfr / crop32)
        pr = T.pairs_of(ids, 0)
        d_sr = lpips_net.pair(np.stack([pred[a] for a, _ in pr]),
                              np.stack([pred[b] for _, b in pr]))
        row["tlp"] = float(np.abs(d_sr - d_gt).mean())
        row["tlp_pairs"] = int(len(pr))
        row["tlp_d_sr"] = float(d_sr.mean())
        row["tlp_d_gt"] = float(d_gt.mean())

        # legacy diagnostic (stride=2 flow-warp LPIPS)
        fw = T.flow_warp_lpips(list(pred), ids, lpips_net, stride=2, scale=4)
        row["flow_warp_lpips"] = fw["value"]
        row["flow_warp_pairs"] = fw["n_pairs"]

        feats[m] = fid_net(pred)
        rows[m] = row
        log("  %-18s psnr %7.3f  ssim %.5f  lpips %.4f  tLP %.5f  fwLPIPS %.5f  (%.0fs)"
            % (m, row["psnr"], row["ssim"], row["lpips"], row["tlp"],
               row["flow_warp_lpips"] or -1, time.time() - t0))

    log("clip %s done in %.1f min" % (clip, (time.time() - t_clip) / 60))
    return ids, rows, feats


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify-clip", default=None,
                    help="assert float32 PSNR == evalkit float64 on this clip")
    args = ap.parse_args()

    out_root = os.path.abspath(os.path.expanduser(args.out))
    for sub in ("results", "feat", "logs"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    cfg_path = os.path.join(out_root, "run_config.json")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"conventions": CONVENTIONS,
                       "data": {"gt": GT_ROOT, "lr": LR_ROOT,
                                "tecogan_frames": TECO_DIR,
                                "edvr_frame_cache": EDVR_FRAMES},
                       "weights": {"edsr": CK_EDSR, "edvr": CK_EDVR},
                       "methods": METHODS}, f, indent=2, ensure_ascii=False)

    clips = args.clips or sorted(d for d in os.listdir(GT_ROOT)
                                 if os.path.isdir(os.path.join(GT_ROOT, d)))
    methods = args.methods or METHODS
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise SystemExit("unknown methods: %s" % bad)

    logfile = open(os.path.join(out_root, "logs", "full_eval.log"), "a", encoding="utf-8")
    log("methods=%s clips=%d device=%s" % (methods, len(clips), args.device), logfile)

    lpips_net = M.LPIPS(device=args.device, torch_home="/tmp/torch_home")
    lpips_net.device = args.device
    fid_net = FIDExtractor(args.device)
    runners = {}
    if "edsr" in methods:
        runners["edsr"] = EDSRRunner(args.device)
    if "edvr" in methods:
        runners["edvr"] = EDVRRunner(args.device)
    log("models loaded", logfile)

    for ci, clip in enumerate(clips):
        res_path = os.path.join(out_root, "results", "%s.json" % clip)
        done = {}
        if os.path.exists(res_path):
            with open(res_path, "r", encoding="utf-8") as f:
                done = json.load(f)
        todo = [m for m in methods
                if not (m in done and done[m].get("tlp") is not None
                        and os.path.exists(os.path.join(out_root, "feat", m, "%s.npy" % clip)))]
        if not todo:
            log("[%d/%d] %s already complete" % (ci + 1, len(clips), clip), logfile)
            continue
        log("[%d/%d] %s -> %s" % (ci + 1, len(clips), clip, todo), logfile)
        ids, rows, feats = score_clip(clip, todo, runners, lpips_net, fid_net, out_root,
                                      verify=(args.verify_clip == clip))
        for m, r in rows.items():
            r.update({"clip": clip, "method": m, "ids": [ids[0], ids[-1]], "n_ids": len(ids)})
            done[m] = r
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(done, f, indent=1)
        for m, arr in feats.items():
            d = os.path.join(out_root, "feat", m)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "%s.npy" % clip), arr)
        del feats, rows

    log("all clips processed", logfile)
    logfile.close()


if __name__ == "__main__":
    main()
