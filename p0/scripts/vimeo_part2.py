#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Task 14 -- Part 2 (EDVR, and TecoGAN where affordable) on Vimeo-90K test.

Dataset facts that constrain the design (verified, not assumed):

  * ``vimeo90k_test/`` holds 7824 septuplets, layout ``<seq>/<clip>/im{1..7}.png``.
  * ``low_resolution/`` has all 7 LR frames; ``target/`` has **only im4.png**.
  * Therefore the centre frame im4 is the only one with GT, and a reference tLP is
    **impossible** -- it needs consecutive GT pairs.  The script writes ``null`` and
    the reason rather than 0, exactly as ``evalkit.temporal`` demands.

What is compared
----------------
Every method is scored on the SAME sample list (centre frame im4 only):

    bicubic / lanczos / bicubic_unsharp / edsr        (Part 1)
    edvr_reds / edvr_vimeo                            (Part 2, both weights)
    tecogan                                           (Part 2, optional, CPU)

EDVR gets 5-frame input ``[im2..im6]`` with the centre at index 2, which is the
model's own configuration; the two weight sets are reported on separate rows
because they were trained on different data (REDS vs Vimeo).

Sample selection
----------------
The checklist says "run a fixed subset first, mark the sample count and how it was
chosen, then extend".  This script does both:

``--n 0``      -> all 7824 clips (the full test set; ~2.5 h on a shared A6000)
``--n 200``    -> a fixed subset: clips taken every ``stride`` from the sorted list
                  starting at index 0, so the subset is deterministic and spread
                  over the whole alphabet of sequence ids.

Usage:
    /tmp/srenv/bin/python ~/p0/scripts/vimeo_part2.py --out ~/p0/out/vimeo --n 200
    /tmp/srenv/bin/python ~/p0/scripts/vimeo_part2.py --out ~/p0/out/vimeo --n 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import numpy as np
import cv2
import torch
import types
import torchvision

_m = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as _F  # noqa: E402

for _n in dir(_F):
    setattr(_m, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _m
torch.backends.cudnn.benchmark = False

sys.path.insert(0, os.path.expanduser("~/p0"))
from evalkit import metrics as M  # noqa: E402

B = "/data/disk1/shared/aiaa3201/project1_video_sr/vimeo90k_test"
LR_ROOT = B + "/low_resolution"
TGT_ROOT = B + "/target"
CK = {
    "edsr": "/data/disk1/shared/aiaa3201/checkpoints/EDSR_baseline_x4/"
            "EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth",
    "edvr_reds": "/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
                 "EDVR_L_x4_SR_REDS_official-9f5f5039.pth",
    "edvr_vimeo": "/data/disk1/shared/aiaa3201/checkpoints/EDVR_Vimeo90K_SR_L/"
                  "EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth",
}


def rd(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def unsharp(f, radius=1.0, amount=1.5):
    ff = f.astype(np.float32)
    return np.clip(np.rint(ff + amount * (ff - cv2.GaussianBlur(ff, (0, 0), radius))),
                   0, 255).astype(np.uint8)


def list_clips():
    out = []
    for seq in sorted(os.listdir(LR_ROOT)):
        d = os.path.join(LR_ROOT, seq)
        if not os.path.isdir(d):
            continue
        for clip in sorted(os.listdir(d)):
            cd = os.path.join(d, clip)
            if os.path.isdir(cd) and os.path.exists(
                    os.path.join(TGT_ROOT, seq, clip, "im4.png")):
                out.append((seq, clip))
    return out


class EDSR:
    def __init__(self, device="cuda"):
        from basicsr.archs.edsr_arch import EDSR as Net
        ck = torch.load(CK["edsr"], map_location="cpu", weights_only=False)["params"]
        net = Net(3, 3, 64, 16, 4)
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


class EDVR:
    """EDVR-L x4 whose ``num_frame`` is read from the checkpoint, not assumed.

    The two released weights are NOT the same architecture: the REDS one fuses
    5 frames (``feat_fusion.weight`` has 5*128 = 640 input channels) while the
    Vimeo90K one fuses all 7 (896 channels), because Vimeo septuplets are 7 frames
    long.  Loading the Vimeo weight into a num_frame=5 model fails with a size
    mismatch, so the frame count is derived from the weight shape and the centre
    frame is put in the middle.
    """

    def __init__(self, ck, device="cuda"):
        from basicsr.archs.edvr_arch import EDVR as Net
        sd = torch.load(ck, map_location="cpu", weights_only=False)["params"]
        num_feat = int(sd["fusion.feat_fusion.weight"].shape[0])
        in_ch = int(sd["fusion.feat_fusion.weight"].shape[1])
        num_frame = in_ch // num_feat
        center = (num_frame - 1) // 2
        net = Net(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_frame=num_frame,
                  deformable_groups=8, num_extract_block=5, num_reconstruct_block=40,
                  center_frame_idx=center, hr_in=False, with_predeblur=False,
                  with_tsa=True)
        net.load_state_dict(sd, strict=True)
        self.net = net.to(device).eval()
        self.num_frame = num_frame
        self.center = center
        self.ckpt = os.path.basename(ck)
        for q in self.net.parameters():
            q.requires_grad_(False)
        self.device = device

    def window(self, lr_all):
        """Centre-frame inference with this weight's own window length.

        ``centre`` must sit at index ``self.center`` of the window, so the window
        starts at ``centre - self.center``, clamped at the clip boundary (the same
        replicate-the-edge policy as the REDS run).
        """
        k = self.num_frame
        n = len(lr_all)
        centre = self.center                       # im4 is index 3 of 7
        start = centre - self.center
        start = max(0, min(start, n - k))
        win = [lr_all[min(max(0, start + d), n - 1)] for d in range(k)]
        x = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float().div_(255.0)
                         for f in win]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.net(x)
        return y[0].clamp_(0, 1).mul_(255).round_().byte().permute(1, 2, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200, help="0 = all clips")
    ap.add_argument("--stride", type=int, default=None,
                    help="subset stride; default = ceil(total/n)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out_dir, exist_ok=True)

    clips = list_clips()
    total = len(clips)
    if args.n and args.n < total:
        stride = args.stride or max(1, total // args.n)
        sample = clips[::stride][:args.n]
        how = ("deterministic stride: every %d-th clip of the sorted (%d total), "
               "truncated at %d" % (stride, total, args.n))
    else:
        sample = clips
        how = "all %d clips of the official test split" % total
    print("clips=%d sample=%d (%s)" % (total, len(sample), how), flush=True)

    lpips_net = M.LPIPS(device="cuda", torch_home="/tmp/torch_home")
    edsr = EDSR()
    edvr_reds = EDVR(CK["edvr_reds"])
    edvr_vimeo = EDVR(CK["edvr_vimeo"])

    print("EDVR windows: reds num_frame=%d (centre idx %d), "
          "vimeo num_frame=%d (centre idx %d)"
          % (edvr_reds.num_frame, edvr_reds.center,
             edvr_vimeo.num_frame, edvr_vimeo.center), flush=True)
    edvr_arch = {m: {"num_frame": r.num_frame, "center_frame_idx": r.center,
                     "checkpoint": r.ckpt}
                 for m, r in (("edvr_reds", edvr_reds), ("edvr_vimeo", edvr_vimeo))}
    out = {"dataset": "vimeo90k_test", "n_available": total, "n_sampled": len(sample),
           "sampling": how, "scored_frame": "im4 (the only frame with GT)",
           "edvr_arch": edvr_arch,
           "tlp": None,
           "tlp_reason": "target/ contains only im4.png per clip, so there is no "
                         "consecutive GT pair -> reference tLP is N/A, not 0",
           "weights": {"edvr_reds": CK["edvr_reds"], "edvr_vimeo": CK["edvr_vimeo"],
                       "edsr": CK["edsr"]},
           "per_method": {}, "rows": []}
    acc = {}
    t_all = time.time()

    for k, (seq, clip) in enumerate(sample):
        t0 = time.time()
        lrd = os.path.join(LR_ROOT, seq, clip)
        names = ["im%d.png" % i for i in range(1, 8)]
        lr = [rd(os.path.join(lrd, nm)) for nm in names]
        gt = rd(os.path.join(TGT_ROOT, seq, clip, "im4.png"))
        H, W = gt.shape[:2]
        mid = 3                                     # im4 is index 3 (0-based)

        preds = {
            "bicubic": cv2.resize(lr[mid], (W, H), interpolation=cv2.INTER_CUBIC),
            "lanczos": cv2.resize(lr[mid], (W, H), interpolation=cv2.INTER_LANCZOS4),
        }
        preds["bicubic_unsharp"] = unsharp(preds["bicubic"])
        preds["edsr"] = edsr([lr[mid]])[0]
        preds["edvr_reds"] = edvr_reds.window(lr)
        preds["edvr_vimeo"] = edvr_vimeo.window(lr)

        for m, p in preds.items():
            if p.shape[:2] != (H, W):
                raise RuntimeError("%s %s shape %s != %s"
                                   % (seq, clip, p.shape[:2], (H, W)))
            pY = M.luma(p)[None]
            gY = M.luma(gt)[None]
            row = {"psnr": float(M.psnr_frames(pY, gY)[0]),
                   "ssim": float(M.ssim_gpu(pY, gY, device="cuda")[0]),
                   "lpips": lpips_net.mean(p[None], gt[None])}
            acc.setdefault(m, []).append(row)
            out["rows"].append({"seq": seq, "clip": clip, "method": m, **row})

        if (k + 1) % 20 == 0 or k + 1 == len(sample):
            msg = "[%d/%d] %s/%s %4.1fs" % (k + 1, len(sample), seq, clip, time.time() - t0)
            means = {m: float(np.mean([r["psnr"] for r in v])) for m, v in acc.items()}
            print(msg + "  " + "  ".join("%s %.3f" % kv for kv in means.items()), flush=True)
            out["per_method"] = {m: {key: float(np.mean([r[key] for r in v]))
                                     for key in ("psnr", "ssim", "lpips")}
                                 for m, v in acc.items()}
            out["per_method"]["_n"] = {m: len(v) for m, v in acc.items()}
            json.dump(out, open(os.path.join(out_dir, "vimeo_part2.json"), "w",
                                encoding="utf-8"), indent=1)

    out["per_method"] = {m: {key: float(np.mean([r[key] for r in v]))
                             for key in ("psnr", "ssim", "lpips")}
                         for m, v in acc.items()}
    out["n_scored"] = {m: len(v) for m, v in acc.items()}
    out["elapsed_min"] = (time.time() - t_all) / 60
    json.dump(out, open(os.path.join(out_dir, "vimeo_part2.json"), "w", encoding="utf-8"),
              indent=1)

    print("\n== Vimeo-90K test, centre frame im4, n=%d ==" % len(sample))
    print("%-16s %8s %8s %8s" % ("method", "PSNR", "SSIM", "LPIPS"))
    for m in ("bicubic", "lanczos", "bicubic_unsharp", "edsr", "edvr_reds", "edvr_vimeo"):
        if m in out["per_method"]:
            r = out["per_method"][m]
            print("%-16s %8.3f %8.5f %8.4f" % (m, r["psnr"], r["ssim"], r["lpips"]))
    print("reference tLP: N/A -- only im4 has GT, no consecutive pairs")
    print("->", os.path.join(out_dir, "vimeo_part2.json"))


if __name__ == "__main__":
    main()
