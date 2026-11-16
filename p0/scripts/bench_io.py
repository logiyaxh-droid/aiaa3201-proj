"""IO cost: reading one clip of GT/LR frames, and what persisting predictions costs."""
import glob
import os
import sys
import time

import cv2
import numpy as np

ROOT = "/data/disk1/shared/aiaa3201/project1_video_sr"


def bench_read(path, n=None):
    files = sorted(glob.glob(os.path.join(path, "*.png")))
    if n:
        files = files[:n]
    t0 = time.perf_counter()
    frames = [cv2.imread(f) for f in files]
    dt = time.perf_counter() - t0
    return len(frames), dt, frames[0].shape if frames else None


def main():
    gt = os.path.join(ROOT, "val_sharp", "000")
    lr = os.path.join(ROOT, "val_sharp_bicubic", "000")
    for name, p in (("GT 720p", gt), ("LR 320x180", lr)):
        for trial in range(2):
            n, dt, sh = bench_read(p)
            print("%-12s %3d frames  %.2f s  -> %.4f s/frame  %s"
                  % (name, n, dt, dt / n, sh))

    # write cost for one method's worth of 720p PNGs
    tmp = "/data/disk1/home/ghe270/p0/out/_io_probe"
    os.makedirs(tmp, exist_ok=True)
    frames = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(gt, "*.png")))[:20]]
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(tmp, "%08d.png" % i), f)
    dt = time.perf_counter() - t0
    sz = sum(os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp))
    print("write 20 PNG: %.2f s -> %.3f s/frame, %.2f MB/frame"
          % (dt, dt / 20, sz / 20 / 1e6))
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    one_method = sz / 20 * 3000 / 1e9
    print("=> persisting one method over 30x100 frames: %.1f GB" % one_method)
    print("=> persisting 12 methods: %.0f GB" % (one_method * 12))


if __name__ == "__main__":
    main()
