"""Throughput benchmark for the tLP / FID estimates. Measures, on real 720p frames:

  1. LPIPS pair cost at batch 25 (what tLP needs, 2 calls per pair).
  2. SSIM/PSNR cost per frame.
  3. Inception-v3 feature cost per frame (what FID needs).
  4. The GT-side reuse saving: d_GT(t) is shared by every method in a clip.

Batches are timed after warm-up; the GPU is shared, so numbers are reported as
a range over repeats rather than a single sample.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/data/disk1/home/ghe270/p0")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")
import types
import torchvision
m = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as F
for n in dir(F):
    setattr(m, n, getattr(F, n))
sys.modules["torchvision.transforms.functional_tensor"] = m

from evalkit import data, metrics, temporal  # noqa: E402

ROOT = "/data/disk1/shared/aiaa3201/project1_video_sr"
CLIP = "000"


def timeit(fn, reps=3):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts, out


def main():
    dev = "cuda"
    print("device:", torch.cuda.get_device_name(0))
    gt_i = data.index_dir(os.path.join(ROOT, "val_sharp", CLIP))
    ids = sorted(gt_i)[:24]
    gt = [data.read_rgb(gt_i[i]) for i in ids]
    gt2 = [data.read_rgb(gt_i[j]) for j in ids[1:]] if len(ids) >= 2 else gt
    n = min(len(gt), len(gt2))
    n = len(gt) - 1          # gt2 is shifted by one frame, so it is 1 shorter
    a, b = np.stack(gt[:n]), np.stack(gt2[:n])
    print("frames:", a.shape)

    net = metrics.LPIPS(device=dev)
    net.pair(a[:2], b[:2])                      # warm-up

    ts, _ = timeit(lambda: net.pair(a[:16], b[:16]), reps=4)
    per_pair = min(ts) / 16
    print("LPIPS  16 pairs: %s s  -> %.4f s/pair" % ([round(x, 3) for x in ts], per_pair))

    ts, _ = timeit(lambda: net.pair(a[:32], b[:32]), reps=3)
    print("LPIPS  32 pairs: %s s  -> %.4f s/pair" % ([round(x, 3) for x in ts], min(ts) / 32))

    pY = np.stack([metrics.luma(x) for x in gt[:n]])
    gY = np.stack([metrics.luma(x) for x in gt2[:n]])
    ts, _ = timeit(lambda: metrics.ssim_gpu(pY, gY, device=dev), reps=3)
    print("SSIM  %d frames: %s s -> %.4f s/frame" % (n, [round(x, 3) for x in ts], min(ts) / n))

    ts, _ = timeit(lambda: metrics.psnr_frames(pY, gY), reps=3)
    print("PSNR  %d frames: %s s -> %.5f s/frame" % (n, [round(x, 4) for x in ts], min(ts) / n))

    # Inception v3 features (FID input side)
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        w = Inception_V3_Weights.IMAGENET1K_V1
        inc = inception_v3(weights=w, transform_input=False, aux_logits=True).to(dev).eval()
        inc.fc = torch.nn.Identity()
        for p in inc.parameters():
            p.requires_grad_(False)

        def feats(x):
            t = torch.from_numpy(np.ascontiguousarray(x)).to(dev).float().div_(255.0)
            t = t.permute(0, 3, 1, 2)
            t = torch.nn.functional.interpolate(t, size=(299, 299), mode="bilinear",
                                                align_corners=False, antialias=True)
            t = (t - 0.5) / 0.5
            with torch.no_grad():
                return inc(t)

        feats(a[:2])
        ts, out = timeit(lambda: feats(a[:32]), reps=3)
        print("Inception 32 frames: %s s -> %.4f s/frame  (feat dim %d)"
              % ([round(x, 3) for x in ts], min(ts) / 32, out.shape[1]))
        ts, _ = timeit(lambda: feats(a[:64]), reps=2)
        print("Inception 64 frames: %s s -> %.4f s/frame" % ([round(x, 3) for x in ts], min(ts) / 64))
    except Exception as e:
        print("Inception unavailable:", repr(e))

    # end-to-end tLP for one clip, 24 frames, with GT-side reuse
    t = temporal.tlp(gt[:n], gt[:n], ids[:n], net)
    ts, _ = timeit(lambda: temporal.tlp(gt[:n], gt[:n], ids[:n], net), reps=2)
    print("tLP  %d frames (%d pairs, 2 LPIPS calls/pair): %.3f s"
          % (n, t["n_pairs"], min(ts)))


if __name__ == "__main__":
    main()
