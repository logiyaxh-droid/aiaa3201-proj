#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Task -- where does EDVR's 5 dB lead over TecoGAN come from?

Hypothesis under test: the gap is a *training-data domain* effect, not model
capacity.  TecoGAN is trained on Vimeo-90K septuplets; the EDVR on the course
shared disk is trained on REDS.  If a Vimeo-trained EDVR also loses ~5 dB on
this REDS evaluation set, the gap follows the training domain.

Control
-------
``edvr_vimeo`` : EDVR-L x4 trained on Vimeo-90K septuplets
                 (``EDVR_L_x4_SR_Vimeo90K_official``), num_frame=7, centre 3.
``edvr``       : the existing REDS-trained EDVR row, read from
                 ``<full>/results/<clip>.json`` -- NOT re-run, so the baseline
                 is bit-identical to the published 30.904 dB.

Everything else is held fixed: the same ``val_sharp_bicubic`` LR frames, the
same 30 clips x 100 frames, the same evalkit PSNR/SSIM/LPIPS/tLP definitions.
The 7-frame window is derived from the weight shape (``fusion.feat_fusion`` has
896 = 7*128 input features), never hard-coded, because the 5-frame REDS build
fails with a size mismatch.

Usage:
    /tmp/srenv/bin/python ~/p0/scripts/edvr_vimeo_ab.py \
        --full ~/p0/out/full --out ~/p0/out/edvr_vimeo_ab [--clips 000 004]
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

_mod = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F  # noqa: E402

for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _mod
torch.backends.cudnn.benchmark = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from evalkit import metrics as M, temporal as T, data as D  # noqa: E402

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
CK_VIMEO = ("/data/disk1/shared/aiaa3201/checkpoints/EDVR_Vimeo90K_SR_L/"
            "EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def rd_rgb(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable frame: %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def psnr_fast(pY, gY):
    out = []
    for i in range(0, len(pY), 20):
        a = pY[i:i + 20].astype(np.float32)
        b = gY[i:i + 20].astype(np.float32)
        d = a - b
        mse = (d * d).mean(axis=tuple(range(1, d.ndim)))
        out.append(10.0 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12)))
    return float(np.concatenate(out).mean())


def load_vimeo_edvr(device="cuda"):
    """Build EDVR-L with the frame count implied by the checkpoint's own shape."""
    from basicsr.archs.edvr_arch import EDVR
    last = None
    for a in range(4):                       # shared disk returns short reads
        try:
            sd = torch.load(CK_VIMEO, map_location="cpu", weights_only=False)["params"]
            break
        except Exception as e:
            last = e
            log("ckpt retry %d/4: %s" % (a + 1, e))
            time.sleep(3)
    else:
        raise RuntimeError("cannot read Vimeo EDVR checkpoint: %r" % last)

    feat_in = sd["fusion.feat_fusion.weight"].shape[1]
    num_frame = feat_in // 128
    center = num_frame // 2
    log("weight implies num_frame=%d (feat_fusion in=%d), centre=%d"
        % (num_frame, feat_in, center))
    net = EDVR(num_in_ch=3, num_out_ch=3, num_feat=128, num_frame=num_frame,
               deformable_groups=8, num_extract_block=5, num_reconstruct_block=40,
               center_frame_idx=center, hr_in=False, with_predeblur=False,
               with_tsa=True)
    net.load_state_dict(sd, strict=True)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, num_frame, center


def run_clip(net, lr, num_frame, center, device):
    n = len(lr)
    out = []
    for c in range(n):
        win = [max(0, min(n - 1, c + d - center)) for d in range(num_frame)]
        x = torch.stack([torch.from_numpy(lr[w]).permute(2, 0, 1).float().div_(255.0)
                         for w in win]).unsqueeze(0).to(device)
        with torch.no_grad():
            y = net(x)
        out.append(y[0].clamp_(0, 1).mul_(255).round_().byte()
                   .permute(1, 2, 0).cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="existing full-eval dir (baseline rows)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out_dir, exist_ok=True)
    full = os.path.abspath(os.path.expanduser(args.full))

    all_clips = sorted(f[:-5] for f in os.listdir(os.path.join(full, "results"))
                       if f.endswith(".json"))
    clips = args.clips or all_clips
    log("clips: %d (%s ... %s)" % (len(clips), clips[0], clips[-1]))

    lpips_net = M.LPIPS(device=args.device, torch_home="/tmp/torch_home")
    net, num_frame, center = load_vimeo_edvr(args.device)

    res = {"control": "EDVR-L x4, Vimeo-90K-trained weights",
           "baseline": "edvr rows read from %s (REDS-trained, not re-run)" % full,
           "weights": {"vimeo": CK_VIMEO},
           "num_frame": num_frame, "center_frame_idx": center,
           "lr": LR_ROOT, "gt": GT_ROOT, "per_clip": {}, "means": {}}
    jpath = os.path.join(out_dir, "edvr_vimeo_ab.json")

    for clip in clips:
        t0 = time.time()
        gt_paths = D.match_clip(os.path.join(GT_ROOT, clip),
                                os.path.join(GT_ROOT, clip))
        ids, gpaths = gt_paths["ids"], gt_paths["gt"]
        gt = [rd_rgb(p) for p in gpaths]
        H, W = gt[0].shape[:2]
        idx = D.index_dir(os.path.join(LR_ROOT, clip))
        lr = [rd_rgb(idx[i]) for i in ids]

        gtY = np.stack([M.luma(g) for g in gt])
        pred = run_clip(net, lr, num_frame, center, args.device)
        pY = np.stack([M.luma(x) for x in pred])

        row = {"n_frames": len(ids),
               "psnr": psnr_fast(pY, gtY),
               "ssim": float(M.ssim_gpu(pY, gtY, device=args.device).mean()),
               "lpips": lpips_net.mean(pred, np.stack(gt))}
        pr = T.pairs_of(ids, 0)
        d_sr = lpips_net.pair(np.stack([pred[a] for a, _ in pr]),
                              np.stack([pred[b] for _, b in pr]))
        d_gt = lpips_net.pair(np.stack([gt[a] for a, _ in pr]),
                              np.stack([gt[b] for _, b in pr]))
        row["tlp"] = float(np.abs(d_sr - d_gt).mean())

        base = json.load(open(os.path.join(full, "results", clip + ".json")))["edvr"]
        row["baseline_edvr_psnr"] = base["psnr"]
        row["delta_vimeo_minus_reds"] = row["psnr"] - base["psnr"]
        res["per_clip"][clip] = row

        log("%s [%4.0fs] vimeo %.3f | reds %.3f | delta %+.3f"
            % (clip, time.time() - t0, row["psnr"], base["psnr"],
               row["delta_vimeo_minus_reds"]))
        json.dump(res, open(jpath, "w", encoding="utf-8"), indent=1)

    keys = ("psnr", "ssim", "lpips", "tlp", "baseline_edvr_psnr",
            "delta_vimeo_minus_reds")
    res["means"] = {k: float(np.mean([r[k] for r in res["per_clip"].values()]))
                    for k in keys}
    json.dump(res, open(jpath, "w", encoding="utf-8"), indent=1)
    m = res["means"]
    log("== mean over %d clips ==" % len(clips))
    log("edvr_vimeo (Vimeo-90K) : %.3f dB" % m["psnr"])
    log("edvr       (REDS)      : %.3f dB" % m["baseline_edvr_psnr"])
    log("delta                 : %+.3f dB" % m["delta_vimeo_minus_reds"])
    log("-> %s" % jpath)


if __name__ == "__main__":
    main()
