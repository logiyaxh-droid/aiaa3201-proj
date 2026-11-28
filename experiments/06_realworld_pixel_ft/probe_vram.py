# -*- coding: utf-8 -*-
"""How much VRAM does one 5-frame 96x96 training step actually need?

The first smoke run OOMed at ~3.5 GB free, but two other users held 44 GB of the
48 GB card, so the failure says nothing about the model's own requirement.  This
script measures it with nothing else running in-process:

  * forward only, batch 1  (cleaning frozen, propagation trainable)
  * forward + backward, batch 1
  * forward + backward, batch 2  (the guide's micro_batch)

and reports peak allocated.  It also times a step, so the 1500-step budget can be
checked against the guide's "35-45 min" claim.
"""
import os
import sys
import time

import torch

HERE = '/data/disk1/home/ghe270/p3/m2b_rbv_ft/scripts'
sys.path.insert(0, HERE)
M2B = os.path.dirname(HERE)
sys.path.insert(0, M2B)
sys.path.insert(0, os.path.join(M2B, 'scripts'))

import warp as W  # noqa: E402
from model_rbv_ft import UARBVSR  # noqa: E402
from train_rbv_ft import Window5, collate  # noqa: E402


def main():
    torch.manual_seed(0)
    model = UARBVSR("cuda")
    ds = Window5(length=100, seed=0)
    sl = slice(W.BR * 4, W.BR * 4 + W.HR_OUT)

    for bs in (1, 2):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = collate([ds[i] for i in range(bs)])
        lr5 = batch["lr5"].cuda()
        gt5 = batch["gt5"].cuda()[:, :, :, sl, sl]

        t0 = time.time()
        pred = []
        for bi in range(bs):
            p = model.forward_trainable(lr5[bi])
            pred.append(p[:, :, sl, sl])
        pred = torch.stack(pred, 0)
        fwd_peak = torch.cuda.max_memory_allocated() / 2 ** 30

        l = ((pred[:, 1:4] - gt5[:, 1:4]) ** 2).mean()
        l.backward()
        step_peak = torch.cuda.max_memory_allocated() / 2 ** 30
        dt = time.time() - t0
        print("batch %d | fwd peak %.2f GB | fwd+bwd peak %.2f GB | %.2fs"
              % (bs, fwd_peak, step_peak, dt), flush=True)
        model.zero_grad(set_to_none=True)
        del pred, l, lr5, gt5
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
