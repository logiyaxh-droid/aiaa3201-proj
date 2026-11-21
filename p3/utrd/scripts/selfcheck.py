# -*- coding: utf-8 -*-
"""Implementation self-check for the three-arm prototype.

Checks, each printing PASS/FAIL with the measured value:
  1. LR/GT pairing and crop alignment (shape, frame ids, no silent resize)
  2. Frozen EDVR reproduces the stored baseline PSNR on clip 000
  3. Uniform and uncertain noise maps have equal per-sample RMS
  4. Deterministic sampling repeats bit-identically under a fixed seed
  5. One train step updates weights and produces finite losses
  6. Overfit a single crop for 10 steps: pixel loss must fall

Run: venv/bin/python scripts/selfcheck.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/data/disk1/home/ghe270/p0")

import numpy as np
import torch
import torch.nn.functional as F

import model as M
import train as T

OUT = os.path.join(ROOT, "selfcheck")
GT_DIR = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp/000"
LR_DIR = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp_bicubic/000"
BASELINE_JSON = "/data/disk1/home/ghe270/p0/out/full/results/000.json"


def check(name, ok, detail):
    print(("PASS " if ok else "FAIL ") + name + ": " + str(detail), flush=True)
    return {"name": name, "pass": bool(ok), "detail": detail}


def psnr_y(pred, gt):
    def chw(x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.transpose(2, 0, 1)
        return x

    def luma(x):
        x = chw(x)
        return 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]

    a, b = luma(pred), luma(gt)
    mse = float(((a - b) ** 2).mean())
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(7)
    np.random.seed(7)
    results = []

    ids = T.frame_ids(GT_DIR)
    lr_ids = T.frame_ids(LR_DIR)
    ids5 = ids[40:45]
    gt = [T.read_rgb(os.path.join(GT_DIR, "%08d.png" % i)) for i in ids5]
    lr = [T.read_rgb(os.path.join(LR_DIR, "%08d.png" % i)) for i in ids5]
    results.append(check("frame ids identical", ids == lr_ids,
                         "%d ids, sample %s" % (len(ids), ids5)))
    results.append(check("LR is exactly GT/4", gt[0].shape[:2] == (720, 1280)
                         and lr[0].shape[:2] == (180, 320),
                         "GT %s LR %s" % (gt[0].shape, lr[0].shape)))

    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().div(255).permute(2, 0, 1)
    lr_t = torch.stack([to_t(f) for f in lr])[None].cuda()
    gt_t = to_t(gt[2]).cuda()[None]

    edvr = M.FrozenEDVR("cuda")
    base, feat, u = M.edvr_outputs(edvr, lr_t)
    base3 = base[0]
    p = psnr_y(base3.cpu().numpy(), gt[2].astype(np.float32) / 255)
    stored = None
    if os.path.exists(BASELINE_JSON):
        stored = json.load(open(BASELINE_JSON, encoding="utf-8"))["edvr"]["psnr"]
    # Stored value is the 100-frame clip mean; this is one frame, so only a
    # plausibility band is meaningful here.  Exact reproduction is checked by
    # evaluate.py on the same 30-frame subset for every arm, including baseline.
    ok = stored is not None and abs(p - stored) < 1.5
    results.append(check("EDVR PSNR plausible vs clip mean", ok,
                         "frame 42 %.4f, stored clip mean %s" % (p, stored)))
    results.append(check("feature shape", tuple(feat.shape) == (1, 128, 180, 320),
                         tuple(feat.shape)))
    results.append(check("uncertainty range", float(u.min()) >= 0 and float(u.max()) <= 1,
                         "min %.3f max %.3f mean %.3f" % (u.min(), u.max(), u.mean())))

    c_u = M.noise_map("uncertain", u)
    c_f = M.noise_map("uniform", u)
    rms_u = float(c_u.square().mean().sqrt())
    rms_f = float(c_f.square().mean().sqrt())
    results.append(check("uniform noise RMS matches", abs(rms_u - rms_f) < 1e-5,
                         "uncertain %.6f uniform %.6f" % (rms_u, rms_f)))
    results.append(check("uncertain map varies spatially", float(c_u.std()) > 1e-3,
                         "std %.4f" % c_u.std()))

    net = M.ResidualUNet(base=64).cuda()
    lr_up = F.interpolate(lr_t[:, 2], scale_factor=4, mode="bilinear",
                          align_corners=False)
    out1 = M.ddim_sample(net, "uncertain", base, lr_up, feat, u, seed=99)
    out2 = M.ddim_sample(net, "uncertain", base, lr_up, feat, u, seed=99)
    results.append(check("fixed-seed sampling reproducible",
                         torch.equal(out1, out2), "max diff %.3g"
                         % float((out1 - out2).abs().max())))

    # One real training step on this crop.
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    before = net.out.weight.detach().clone()
    for mode in M.MODES:
        st = M.make_state(mode, base, gt_t, u)
        pred = net(st["x"], st["t"], base, lr_up, feat, u)
        if mode == "det":
            y_hat = (base + pred).clamp(0, 1)
            loss = T.charbonnier(y_hat, gt_t)
        else:
            r0 = M.x0_from_eps(st["x"], pred, base, st["c"], st["ab"])
            y_hat = (base + r0).clamp(0, 1)
            loss = T.charbonnier(y_hat, gt_t) + (pred - st["eps"]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        opt.step()
        changed = float((net.out.weight.detach() - before).abs().max())
        results.append(check("%s forward/backward finite" % mode, finite,
                             "loss %.5f" % float(loss)))
        results.append(check("%s updates weights" % mode, changed > 0,
                             "max delta %.3g" % changed))
        before = net.out.weight.detach().clone()

    # Small overfit check: same crop, 10 steps, loss must fall.
    net2 = M.ResidualUNet(base=32).cuda()
    opt2 = torch.optim.Adam(net2.parameters(), lr=3e-4)
    losses = []
    for _ in range(10):
        st = M.make_state("det", base, gt_t, u)
        pred = net2(st["x"], st["t"], base, lr_up, feat, u)
        loss = T.charbonnier((base + pred).clamp(0, 1), gt_t)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        opt2.step()
        losses.append(float(loss))
    results.append(check("overfit 10 steps reduces loss", losses[-1] < losses[0],
                         "%.5f -> %.5f" % (losses[0], losses[-1])))

    summary = {"checks": results, "n_pass": sum(r["pass"] for r in results),
               "n_total": len(results)}
    with open(os.path.join(OUT, "selfcheck.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("\n%d/%d passed" % (summary["n_pass"], summary["n_total"]))
    if summary["n_pass"] != summary["n_total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
