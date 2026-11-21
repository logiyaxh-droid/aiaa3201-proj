#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Part 3 / Direction C step 1 -- dump EDVR's own confidence signals.

Direction C asks for a pixel-wise uncertainty map that decides where to trust a
"steady" reconstruction and where to fall back to a conservative one. EDVR
already computes two such signals internally and then throws them away:

  * temporal attention  corr_prob (b,t,h,w)  -- sigmoid of the feature
    correlation between the centre frame and each neighbour. Low value = the
    neighbouring frames carry no usable information for that pixel (occlusion,
    motion beyond the deformable window, texture the alignment could not lock).
  * spatial attention   attn (b,1,h,w)       -- the TSA gate applied to the fused
    feature. Low value = the fusion stage damped this region.

Neither is a calibrated uncertainty in the Bayesian sense, but both are free
(no extra forward pass) and they are *exactly* the places where the model had to
guess. This script writes them out next to the HR reconstruction so Direction C
can be tuned offline without re-running inference.

Output per clip under OUT:
  {clip}_attn.npz     corr_mean (h,w) float16, attn (h,w) float16   [LR grid]
  {clip}_heat.png     side-by-side visualisation, 3 panels upscaled x4
Run:  python p3_attn_dump.py 000 002 009
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, types, glob, time
import numpy as np
import torch
import torch.nn.functional as Fn
import cv2

import torchvision
_m = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as F
for _n in dir(F):
    setattr(_m, _n, getattr(F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = _m

from basicsr.archs.edvr_arch import EDVR, TSAFusion

torch.backends.cudnn.benchmark = True

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
CK = ("/data/disk1/shared/aiaa3201/checkpoints/EDVR_REDS_SR_L/"
      "EDVR_L_x4_SR_REDS_official-9f5f5039.pth")
OUT = "/data/disk2/home/ghe270/p3/attn"

# ------------------------------------------------------------------ hook
_STORE = {}


def _tsa_forward(self, aligned_feat):
    """Verbatim copy of TSAFusion.forward with the two gates recorded."""
    b, t, c, h, w = aligned_feat.size()
    embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx].clone())
    embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w)).view(b, t, -1, h, w)
    corr_l = []
    for i in range(t):
        corr = torch.sum(embedding[:, i] * embedding_ref, 1)
        corr_l.append(corr.unsqueeze(1))
    corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))          # (b,t,h,w)
    _STORE["corr"] = corr_prob.detach()

    corr_e = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
    corr_e = corr_e.contiguous().view(b, -1, h, w)
    aligned_feat = aligned_feat.view(b, -1, h, w) * corr_e

    feat = self.lrelu(self.feat_fusion(aligned_feat))

    attn = self.lrelu(self.spatial_attn1(aligned_feat))
    attn_max = self.max_pool(attn)
    attn_avg = self.avg_pool(attn)
    attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
    attn_level = self.lrelu(self.spatial_attn_l1(attn))
    attn_max = self.max_pool(attn_level)
    attn_avg = self.avg_pool(attn_level)
    attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
    attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
    attn_level = self.upsample(attn_level)
    attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
    attn = self.lrelu(self.spatial_attn4(attn))
    attn = self.upsample(attn)
    attn = self.spatial_attn5(attn)
    attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
    attn = torch.sigmoid(attn)
    _STORE["attn"] = attn.detach()

    return feat * attn * 2 + attn_add


TSAFusion.forward = _tsa_forward


# ------------------------------------------------------------------ model
def _load_ck(path, tries=6):
    import shutil
    for a in range(tries):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)["params"]
        except Exception as e:
            print("ckpt retry %d/%d: %s" % (a + 1, tries, e), flush=True)
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


def to_u8(t):
    return t.clamp_(0, 1).mul_(255).round_().byte().permute(1, 2, 0).cpu().numpy()


def heat(m, lo=0.0, hi=1.0, size=None):
    """float [0,1] map -> BGR colour image (COLORMAP_TURBO, higher = hotter)."""
    a = np.clip((m - lo) / max(1e-6, hi - lo), 0, 1)
    a = (a * 255).astype(np.uint8)
    if size is not None:
        a = cv2.resize(a, size, interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(a, cv2.COLORMAP_TURBO)


if __name__ == "__main__":
    clips = sys.argv[1:] or ["000"]
    os.makedirs(OUT, exist_ok=True)
    for clip in clips:
        files = sorted(glob.glob(os.path.join(GT_ROOT, clip, "*.png")))
        n = len(files)
        t0 = time.time()
        C, A, HR = [], [], []
        for c in range(n):
            win = [max(0, min(n - 1, c + k - 2)) for k in range(5)]
            x = torch.stack([rd("%s/%s/%08d.png" % (LR_ROOT, clip, w)) for w in win]).unsqueeze(0).cuda()
            with torch.no_grad():
                y = model(x)
            C.append(_STORE["corr"].mean(1)[0].float().cpu().numpy())
            A.append(_STORE["attn"][0, 0].float().cpu().numpy())
            HR.append(to_u8(y[0].clone()))

        C = np.stack(C).astype(np.float16)
        A = np.stack(A).astype(np.float16)
        np.savez_compressed(os.path.join(OUT, "%s_attn.npz" % clip), corr=C, attn=A)

        # 3 panels: reconstruction | temporal attention | spatial attention (frame 50)
        f = n // 2
        hr = cv2.cvtColor(HR[f], cv2.COLOR_RGB2BGR)
        H, W = hr.shape[:2]
        cimg = heat(C[f].astype(np.float32), C.min(), C.max(), (W, H))
        aimg = heat(A[f].astype(np.float32), A.min(), A.max(), (W, H))
        pad = np.full((H, 8, 3), 255, np.uint8)
        cv2.imwrite(os.path.join(OUT, "%s_heat.png" % clip),
                    np.hstack([hr, pad, cimg, pad, aimg]))
        print("%s frames=%d  corr[%.3f,%.3f] mean %.3f | attn[%.3f,%.3f] mean %.3f | %.0fs"
              % (clip, n, C.min(), C.max(), C.mean(), A.min(), A.max(), A.mean(),
                 time.time() - t0), flush=True)
    print("wrote ->", OUT, flush=True)
