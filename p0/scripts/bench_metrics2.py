"""Second-pass benchmark: the numbers that actually drive the full-run estimate.

  * PSNR at its current float64 cost vs a float32 accumulation (same value).
  * LPIPS at the batch size the real run will use (99 pairs of 720p).
  * One clip's flow-fusion cost, measured end to end rather than derived.
"""
import os
import sys
import time
import types

import numpy as np
import torch
import torchvision

m = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms import functional as F
for n in dir(F):
    setattr(m, n, getattr(F, n))
sys.modules["torchvision.transforms.functional_tensor"] = m

sys.path.insert(0, "/data/disk1/home/ghe270/p0")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")
from evalkit import data, metrics, temporal  # noqa: E402

ROOT = "/data/disk1/shared/aiaa3201/project1_video_sr"


def timeit(fn, reps=2):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts), ts


def main():
    import cv2
    gt_i = data.index_dir(os.path.join(ROOT, "val_sharp", "000"))
    ids = sorted(gt_i)
    gt = [data.read_rgb(gt_i[i]) for i in ids]
    gY = np.stack([metrics.luma(x) for x in gt])
    pY = np.stack([metrics.luma(x) for x in gt])
    print("Y stack:", gY.shape, gY.dtype)

    t_old, _ = timeit(lambda: metrics.psnr_frames(pY, gY), reps=2)
    print("PSNR current (float64) 100 frames: %.3f s -> %.4f s/frame"
          % (t_old, t_old / 100))

    def psnr_f32(a, b):
        acc = np.zeros(len(a), np.float64)
        for i in range(len(a)):
            d = a[i].astype(np.float32) - b[i].astype(np.float32)
            d *= d
            acc[i] = d.mean(dtype=np.float64)
        return 10.0 * np.log10(255.0 ** 2 / np.maximum(acc, 1e-12))

    t_new, _ = timeit(lambda: psnr_f32(pY, gY), reps=2)
    v_old = metrics.psnr_frames(pY, gY)
    v_new = psnr_f32(pY, gY)
    print("PSNR float32 streaming 100 frames: %.3f s -> %.4f s/frame  (max diff %.2e)"
          % (t_new, t_new / 100, np.abs(v_old - v_new).max()))

    t_ss, _ = timeit(lambda: metrics.ssim_gpu(pY, gY), reps=2)
    print("SSIM 100 frames: %.3f s -> %.4f s/frame" % (t_ss, t_ss / 100))

    net = metrics.LPIPS()
    net.pair(np.stack(gt[:2]), np.stack(gt[:2]))
    t_lp, _ = timeit(lambda: net.pair(np.stack(gt[:99]), np.stack(gt[1:100])), reps=3)
    print("LPIPS 99 pairs (1 call): %.3f s -> %.4f s/pair" % (t_lp, t_lp / 99))

    t_tl, _ = timeit(lambda: temporal.tlp(gt, gt, ids, net), reps=2)
    print("tLP full clip, 99 pairs, 2 LPIPS calls: %.3f s" % t_tl)

    # one clip of flow fusion, as the real run does it
    lr_i = data.index_dir(os.path.join(ROOT, "val_sharp_bicubic", "000"))
    lr = [data.read_rgb(lr_i[i]) for i in ids]
    H, W = gt[0].shape[:2]
    bic = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr]
    gbic = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in bic]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    def fuse(kernel):
        h, w = bic[0].shape[:2]
        wt = np.asarray(kernel, np.float32)
        wt /= wt.sum()
        k = len(wt) // 2
        n = len(bic)
        cache = {}

        def get(t, j):
            if j == t:
                return bic[t].astype(np.float32)
            if (t, j) not in cache:
                cache[(t, j)] = temporal.warp_onto(bic[j], gbic[t], gbic[j], gx, gy).astype(np.float32)
            return cache[(t, j)]

        out = np.zeros((n, h, w, 3), np.float32)
        for t in range(n):
            acc = np.zeros((h, w, 3), np.float32)
            for d, w_i in enumerate(wt):
                acc += w_i * get(t, min(max(t + d - k, 0), n - 1))
            out[t] = acc
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    t5, _ = timeit(lambda: fuse((1, 2, 3, 2, 1)), reps=1)
    t3, _ = timeit(lambda: fuse((1, 2, 1)), reps=1)
    print("flow fusion 5-tap, 100 frames: %.1f s" % t5)
    print("flow fusion 3-tap, 100 frames: %.1f s" % t3)


if __name__ == "__main__":
    main()
