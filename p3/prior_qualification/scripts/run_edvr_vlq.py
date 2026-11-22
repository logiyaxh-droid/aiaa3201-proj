# -*- coding: utf-8 -*-
"""Run EDVR on VideoLQ with the self-consistent x4 protocol.

VideoLQ has no GT.  For a x4 model the only well-posed comparison is:
    native frame -> /4 -> EDVR(x4) -> compare with the native frame.
If a model destroys information the native frame already had, self_psnr drops.

Small clips (320x240 -> 80x60) are kept as-is: EDVR accepts any /4 size, and
resizing would change what we are measuring.  The report must state that very
small inputs are outside EDVR's training distribution.

Output: outputs/vlq/<name>/<clip>/<frame>.png  (native size)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2
import torch

ROOT = "/data/disk1/home/ghe270/p3/prior_qualification"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p3/utrd/scripts")
sys.path.insert(0, "/data/disk1/home/ghe270/p0")

import model as UM  # noqa: E402

SRC = os.path.join(ROOT, "data/vlq_eval")
FT_CK = ("/data/disk1/home/ghe270/p3/quality_metrics/fine_tune/runs/"
         "adv_lr3e5_d003_a001/step_000900.pth")


def build(finetuned):
    net = UM.FrozenEDVR("cuda")
    if finetuned:
        ck = torch.load(FT_CK, map_location="cpu", weights_only=False)
        net.net.load_state_dict(ck["model"], strict=True)
        net.net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="edvr")
    ap.add_argument("--finetuned", default="false")
    args = ap.parse_args()
    net = build(args.finetuned.lower() in ("1", "true", "yes"))

    out_root = os.path.join(ROOT, "outputs/vlq", args.name)
    clips = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
    t0 = time.time()
    done = 0
    for clip in clips:
        files = sorted(glob.glob(os.path.join(SRC, clip, "*.png")))
        od = os.path.join(out_root, clip)
        os.makedirs(od, exist_ok=True)
        # EDVR uses five consecutive frames; these are 10-frame samples, so pass
        # the real neighbouring samples rather than repeating one frame.
        for i, p in enumerate(files):
            im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            h, w = im.shape[:2]
            # keep /4 integer and >= 8 px for the deformable pyramid
            w4, h4 = max(8, (w // 4) - (w // 4) % 2), max(8, (h // 4) - (h // 4) % 2)
            lr = cv2.resize(im, (w4, h4), interpolation=cv2.INTER_CUBIC)
            x = torch.from_numpy(lr).permute(2, 0, 1).float().div(255)
            x = x[None].repeat(5, 1, 1, 1)[None].cuda()
            with torch.no_grad():
                y, _ = net(x)
            out = y[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(od, os.path.basename(p)),
                        cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            done += 1
        if clip in ("000", "010", "020", "030", "040", "049"):
            print("%s %d frames %.1fs" % (clip, len(files), time.time() - t0), flush=True)
    print("%s done %d frames %.1fs" % (args.name, done, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
