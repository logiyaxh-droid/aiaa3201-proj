"""Throughput of the two expensive parts of the Part-1 pipeline: EDSR inference
and the optical-flow fusion. Measured on real 320x180 LR / 1280x720 HR frames.
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
from evalkit import data, temporal  # noqa: E402

ROOT = "/data/disk1/shared/aiaa3201/project1_video_sr"
CK = ("/data/disk1/shared/aiaa3201/checkpoints/EDSR_baseline_x4/"
      "EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth")
DEV = "cuda"


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
    import cv2
    from basicsr.archs.edsr_arch import EDSR

    lr_i = data.index_dir(os.path.join(ROOT, "val_sharp_bicubic", "000"))
    ids = sorted(lr_i)[:40]
    lr = [data.read_rgb(lr_i[i]) for i in ids]
    print("LR:", lr[0].shape, " frames:", len(lr))

    ck = torch.load(CK, map_location="cpu", weights_only=False)["params"]
    edsr = EDSR(3, 3, 64, 16, 4)
    edsr.load_state_dict(ck, strict=True)
    edsr = edsr.to(DEV).eval()
    for p in edsr.parameters():
        p.requires_grad_(False)

    def run(n, bs):
        out = []
        for i in range(0, n, bs):
            x = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float().div_(255.0)
                             for f in lr[i:i + bs]]).to(DEV)
            with torch.no_grad():
                y = edsr(x)
            y = y.clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).cpu().numpy()
            out.extend(y[k] for k in range(y.shape[0]))
        return out

    run(4, 4)
    for bs in (4, 8):
        ts, o = timeit(lambda: run(32, bs), reps=3)
        print("EDSR bs=%d 32 frames: %s s -> %.3f s/frame  (out %s)"
              % (bs, [round(x, 2) for x in ts], min(ts) / 32, o[0].shape))

    # flow: Farneback at 1/4 res on 320x180 -> flow for 1280x720 warp
    H, W = 720, 1280
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    hr = [cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC) for x in lr[:12]]
    grays = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in hr]

    def flow_pairs(n):
        for t in range(1, n):
            temporal.fb_target_to_source(grays[t], grays[t - 1], scale=4)

    ts, _ = timeit(lambda: flow_pairs(11), reps=2)
    per_flow = min(ts) / 10
    print("Farneback(1/4 res) per pair: %.3f s  [10 pairs: %s s]"
          % (per_flow, [round(x, 2) for x in ts]))

    def warp_one(n):
        for t in range(1, n):
            temporal.warp_onto(hr[t - 1], grays[t], grays[t - 1], gx, gy, scale=4)

    ts, _ = timeit(lambda: warp_one(11), reps=2)
    per_warp = min(ts) / 10
    print("flow+warp per pair: %.3f s  (flow %.3f + warp+remap %.3f)"
          % (per_warp, per_flow, per_warp - per_flow))

    # full flow_temporal_avg for a 5-tap kernel over n frames: ~n*(taps-1) warps
    def fta(n):
        temporal.flow_temporal_avg if hasattr(temporal, "flow_temporal_avg") else None

    print("\nDerived: 100-frame clip, 5-tap flow fusion needs about %d flow estimates "
          "-> %.1f s/clip, 30 clips -> %.0f min"
          % (99 * 4, 99 * 4 * per_warp, 99 * 4 * per_warp * 30 / 60))


if __name__ == "__main__":
    main()
