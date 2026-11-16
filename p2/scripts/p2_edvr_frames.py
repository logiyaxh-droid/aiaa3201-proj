#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-run EDVR for a handful of clips only, keeping the frames this time.

EDVR output is 1.5 GB per clip at 1000 frames, so the full 30-clip run streams
frames to metrics and discards them (see p2_edvr.py). For the report we still
need a few real EDVR frames for (a) the enlarged patch figure and (b) the
4-panel comparison videos, so this script regenerates just those clips and
writes the frames to /tmp/p2/edvr_frame/<clip>/.
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, types, json, glob, time
import numpy as np
import torch
import cv2

import torchvision
_mod = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as _F
for _n in dir(_F):
    setattr(_mod, _n, getattr(_F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = _mod

from basicsr.archs.edvr_arch import EDVR

torch.backends.cudnn.benchmark = True

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
LR_ROOT = B + "/val_sharp_bicubic"
CK = ("/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
      "EDVR_L_x4_SR_REDS_official-9f5f5039.pth")
OUT = "/tmp/p2/edvr_frame"


def _load_ck(path, tries=6):
    import shutil
    for a in range(tries):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)["params"]
        except Exception as e:
            print("ckpt retry %d: %s" % (a + 1, e), flush=True)
            time.sleep(4)
    dst = "/tmp/edvr_ck.pth"
    shutil.copyfile(path, dst)
    return torch.load(dst, map_location="cpu", weights_only=False)["params"]


ck = _load_ck(CK)
model = EDVR(num_in_ch=3, num_out_ch=3, num_feat=128, num_frame=5,
             deformable_groups=8, num_extract_block=5, num_reconstruct_block=40,
             center_frame_idx=2, hr_in=False, with_predeblur=False, with_tsa=True)
model.load_state_dict(ck, strict=True)
model = model.cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)


def rd(p):
    a = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


if __name__ == "__main__":
    clips = sys.argv[1:] or ["000"]
    for clip in clips:
        d = os.path.join(OUT, clip)
        os.makedirs(d, exist_ok=True)
        files = sorted(glob.glob(os.path.join(LR_ROOT, clip, "*.png")))
        n = len(files)
        t0 = time.time()
        for c in range(n):
            win = [max(0, min(n - 1, c + k - 2)) for k in range(5)]
            x = torch.stack([rd("%s/%s/%08d.png" % (LR_ROOT, clip, w)) for w in win]).unsqueeze(0).cuda()
            with torch.no_grad():
                y = model(x)
            y = y[0].clamp_(0, 1).mul_(255).round_().byte().permute(1, 2, 0).cpu().numpy()
            cv2.imwrite(os.path.join(d, "%08d.png" % c),
                        cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
        print("%s %d frames %.1fs" % (clip, n, time.time() - t0), flush=True)
    print("done ->", OUT)
