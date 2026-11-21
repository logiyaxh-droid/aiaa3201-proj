"""Direction C, third cut -- TTA ensembling: the variant that has actual headroom.

Why the previous two cuts failed, in one line: both took BasicVSR++ as the base
and tried to borrow *pixels* from a weaker model (Real-ESRGAN is 7.9 dB down on
bicubic degradation), so every routing weight cost PSNR and bought nothing
perceptually. The destination was dominated, not the router.

This cut changes the destination. BasicVSR++ is exactly equivariant to a
horizontal flip, so running it twice -- once as-is, once on flipped frames -- and
averaging gives an ensemble with smaller variance, and the *spread between the two
members* is a genuine per-pixel uncertainty with no external model involved:

    Y_tta = (Y(x) + flip(Y(flip(x)))) / 2
    U_tta = |Y(x) - flip(Y(flip(x)))|          (exact epistemic-style spread)

Reported: PSNR/LPIPS of the single pass vs the TTA ensemble, and
Spearman(U_tta, |Y_tta - GT|) to show the uncertainty is calibrated -- that
last number is what turns "we routed pixels around" into "we can say where the
model is unreliable", which is what the report needs either way.

Run: python p3_tta.py 000 002 009
"""
import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

import sys, types, glob, time, re, json
import numpy as np
import torch
import cv2

import torchvision
_m = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as F
for _n in dir(F):
    setattr(_m, _n, getattr(F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = _m

from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
from scipy.stats import spearmanr
import lpips

torch.backends.cudnn.benchmark = True

B = "/data/disk1/shared/aiaa3201/project1_video_sr"
GT_ROOT = B + "/val_sharp"
LR_ROOT = B + "/val_sharp_bicubic"
CK = "/data/disk2/home/ghe270/p3/ckpt/bvsrpp_reds4.pth"
SPY = "/data/disk2/home/ghe270/p3/ckpt/spynet_bsr.pth"
OUT = "/data/disk2/home/ghe270/p3"

sd = torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]
new = {}
for k, v in sd.items():
    if k == "step_counter":
        continue
    k2 = k[len("generator."):] if k.startswith("generator.") else k
    if k2.startswith("spynet."):
        continue
    k2 = k2.replace("upsample1.upsample_conv", "upconv1").replace("upsample2.upsample_conv", "upconv2")
    new[k2] = v
model = BasicVSRPlusPlus(mid_channels=64, num_blocks=7, is_low_res_input=True, spynet_path=SPY)
miss, unexp = model.load_state_dict(new, strict=False)
assert not [x for x in miss if not x.startswith("spynet.")] and not unexp, (miss, unexp)
model = model.cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)

net = lpips.LPIPS(net="alex").cuda().eval()
for p in net.parameters():
    p.requires_grad_(False)

CHUNK = 40


def to_t(bgr):
    return torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).cuda().float() \
        .div_(127.5).sub_(1.0).permute(2, 0, 1).unsqueeze(0)


def lp1(a, b):
    with torch.no_grad():
        return float(net(to_t(a), to_t(b)).reshape(-1)[0])


def ly(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]


def psnr1(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return 10.0 * np.log10(255.0 ** 2 / max((d * d).mean(), 1e-12))


def infer(lq):
    """lq: (n,h,w,3) uint8 -> (n,h,w,3) uint8, chunked to bound GPU memory."""
    n = len(lq)
    out = []
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        x = torch.from_numpy(np.ascontiguousarray(lq[a:b])).float().div_(255.0).permute(0, 3, 1, 2).unsqueeze(0).cuda()
        with torch.no_grad():
            y = model(x)
        out.append(y[0].clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).cpu().numpy())
        del x, y
        torch.cuda.empty_cache()
    return np.concatenate(out)


def run(clip):
    lrf = sorted(glob.glob(os.path.join(LR_ROOT, clip, "*.png")))
    gtf = sorted(glob.glob(os.path.join(GT_ROOT, clip, "*.png")))
    n = min(len(lrf), len(gtf))
    lq = np.stack([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in lrf[:n]])
    gt = np.stack([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in gtf[:n]])

    t0 = time.time()
    y1 = infer(lq)
    y2 = np.ascontiguousarray(infer(np.ascontiguousarray(lq[:, :, ::-1]))[:, :, ::-1])
    tta = ((y1.astype(np.uint16) + y2.astype(np.uint16) + 1) // 2).astype(np.uint8)

    e1 = np.stack([np.abs(ly(y1[i]).astype(np.float32) - ly(gt[i]).astype(np.float32)) for i in range(n)])
    et = np.stack([np.abs(ly(tta[i]).astype(np.float32) - ly(gt[i]).astype(np.float32)) for i in range(n)])
    U = np.stack([np.abs(ly(y1[i]).astype(np.float32) - ly(y2[i]).astype(np.float32)) for i in range(n)])

    sub = np.random.RandomState(0).choice(et.size, min(400000, et.size), replace=False)
    rho = float(spearmanr(et.ravel()[sub], U.ravel()[sub]).correlation)

    gty = np.stack([ly(g) for g in gt])
    r = dict(
        clip=clip, frames=int(n), sec=round(time.time() - t0, 1),
        psnr_single=round(float(np.mean([psnr1(ly(y1[i]), ly(gt[i])) for i in range(n)])), 3),
        psnr_tta=round(float(np.mean([psnr1(ly(tta[i]), ly(gt[i])) for i in range(n)])), 3),
        lpips_single=round(float(np.mean([lp1(y1[i][:, :, ::-1], gt[i][:, :, ::-1]) for i in range(n)])), 4),
        lpips_tta=round(float(np.mean([lp1(tta[i][:, :, ::-1], gt[i][:, :, ::-1]) for i in range(n)])), 4),
        spearman_tta=round(rho, 4),
        spread_mean=round(float(U.mean()), 3))
    return r


if __name__ == "__main__":
    res = {}
    for clip in (sys.argv[1:] or ["000"]):
        r = run(clip)
        res[clip] = r
        print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(OUT, "tta.json"), "w"), indent=2)
    print("OK")
