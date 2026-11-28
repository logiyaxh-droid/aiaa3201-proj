# -*- coding: utf-8 -*-
"""M2-1 dataset: 5-frame LR windows -> 96x96 crop -> (B_t, B_prev, flip pair, GT).

Design notes that are not obvious
---------------------------------
* The 5-frame window is what RealBasicVSR needs for its bidirectional
  propagation; the target is the *centre* frame (index 2), matching the way RBV
  is evaluated (frame 24 of a 20..28 span is the centre of a 5-window).
* Crops are taken at 96x96 from the 320x180 LR (= 384x384 HR), and the GT
  counterpart is the matching 384x384 region, from which the model later takes
  the centre 256x256.  Aligning the crop *before* RBV is what makes the GT crop
  line up: LR pixel p corresponds to GT pixels [4p, 4p+4).
* Two RNGs are kept apart on purpose: `self.rng` drives sampling (which clip /
  frame / crop), `self.aug` drives augmentation.  Sharing one made resume-time
  ordering matter in an earlier project.
* All randomness is seeded from the worker id so DataLoader workers are
  reproducible without `worker_init_fn` surprises.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

SRC_GT = "/data/disk1/shared/aiaa3201/project1_video_sr/train_sample_sharp"
LR_ROOT = "/data/disk1/home/ghe270/p3/m2_stageA/data/train_lr"
LR_CROP = 96
WIN = 5


def _read_rgb(p):
    a = cv2.imread(p)
    if a is None:
        raise RuntimeError("unreadable %s" % p)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


class WindowDataset(Dataset):
    """`length` is virtual: samples are drawn independently, not epoch-wise.

    That keeps the step budget independent of dataset size, which is what the
    1,500-step Pilot and 5,000-step main stage are specified in.
    """

    def __init__(self, tiers=("v0", "v1", "v2"), length=100000, seed=0,
                 crop=LR_CROP, win=WIN, aug=True, clips=None, lr_root=LR_ROOT,
                 gt_root=SRC_GT):
        self.tiers = list(tiers)
        self.length = int(length)
        self.crop = int(crop)
        self.win = int(win)
        self.aug = bool(aug)
        self.lr_root = lr_root
        self.gt_root = gt_root

        avail = sorted(os.listdir(os.path.join(lr_root, self.tiers[0])))
        self.clips = [c for c in (clips or avail)
                      if os.path.isdir(os.path.join(lr_root, self.tiers[0], c))]
        if not self.clips:
            raise RuntimeError("no clips under %s" % lr_root)
        # frame ids are 8-digit in both the LR tree and the GT tree
        self.nframes = len([f for f in
                            os.listdir(os.path.join(lr_root, self.tiers[0],
                                                    self.clips[0]))
                            if f.lower().endswith(".png")])
        if self.nframes < self.win:
            raise RuntimeError("only %d frames, need >= %d"
                               % (self.nframes, self.win))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.aug_rng = np.random.default_rng(self.seed + 7919)

    def __len__(self):
        return self.length

    def _ids(self, centre):
        lo = centre - self.win // 2
        hi = lo + self.win
        lo = max(0, min(lo, self.nframes - self.win))
        return list(range(lo, lo + self.win))

    def __getitem__(self, index):
        # sampling RNG: deterministic per index, independent of call order
        r = np.random.default_rng(self.seed * 1000003 + index)
        clip = self.clips[int(r.integers(len(self.clips)))]
        tier = self.tiers[int(r.integers(len(self.tiers)))]
        centre = int(r.integers(self.win // 2, self.nframes - self.win // 2))

        ids = self._ids(centre)
        lr_dir = os.path.join(self.lr_root, tier, clip)
        lr = np.stack([_read_rgb(os.path.join(lr_dir, "%08d.png" % i))
                       for i in ids])                      # [5,180,320,3]
        gt_dir = os.path.join(self.gt_root, clip)
        tgt_i = ids[self.win // 2]
        gt = _read_rgb(os.path.join(gt_dir, "%08d.png" % tgt_i))   # [720,1280,3]
        prev_i = ids[self.win // 2 - 1]
        gt_prev = _read_rgb(os.path.join(gt_dir, "%08d.png" % prev_i))

        h, w = lr.shape[1:3]
        c = self.crop
        y = int(r.integers(0, h - c + 1))
        x = int(r.integers(0, w - c + 1))
        lr = lr[:, y:y + c, x:x + c]
        gt = gt[y * 4:(y + c) * 4, x * 4:(x + c) * 4]
        gt_prev = gt_prev[y * 4:(y + c) * 4, x * 4:(x + c) * 4]

        # augmentation: horizontal flip, applied to every frame AND the GT so
        # the pair stays aligned; the flip branch of U_1 is unaffected because
        # it is computed inside the model on its own copy.
        if self.aug:
            a = np.random.default_rng(self.seed * 7331 + index)
            if a.random() < 0.5:
                lr = lr[:, :, ::-1]
                gt = gt[:, :, ::-1]
                gt_prev = gt_prev[:, :, ::-1]
            if a.random() < 0.5:
                # the crop is square, so transposing keeps every shape valid
                lr = np.ascontiguousarray(np.transpose(lr, (0, 2, 1, 3)))
                gt = np.ascontiguousarray(np.transpose(gt, (1, 0, 2)))
                gt_prev = np.ascontiguousarray(np.transpose(gt_prev, (1, 0, 2)))
        def to_t(a):
            return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0

        out = {
            "lr5": torch.stack([to_t(f) for f in lr]),      # [5,3,96,96]
            "gt": to_t(gt),                                  # [3,384,384]
            "gt_prev": to_t(gt_prev),
            "lr_t": to_t(lr[self.win // 2]),                 # [3,96,96]
            "lr_prev": to_t(lr[self.win // 2 - 1]),
            "meta": {"clip": clip, "tier": tier, "frame": int(tgt_i)},
        }
        return out


def collate(batch):
    return {
        "lr5": torch.stack([b["lr5"] for b in batch]),
        "gt": torch.stack([b["gt"] for b in batch]),
        "gt_prev": torch.stack([b["gt_prev"] for b in batch]),
        "lr_t": torch.stack([b["lr_t"] for b in batch]),
        "lr_prev": torch.stack([b["lr_prev"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }
