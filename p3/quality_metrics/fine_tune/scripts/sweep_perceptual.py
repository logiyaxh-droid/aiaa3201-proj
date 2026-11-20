# -*- coding: utf-8 -*-
"""Perceptual-loss weight sweep, single variable at a time.

Design decisions that are easy to get wrong here:

* A pixel-only run at the SAME learning rate and step count is trained first.
  Fine-tuning from a converged checkpoint at a non-trivial learning rate damages
  the model on its own (measured in the pilot: 400 steps at lr 1e-4 cost 0.23 dB
  of PSNR and 0.003 of DISTS).  Comparing a perceptual run against the original
  weights would therefore credit the perceptual term with the fine-tuning damage.
  Every reported delta is relative to the pixel-only control.
* Only DISTS is swept here.  The adversarial term is added later, after a
  perceptual weight is chosen, so the two effects never move together.
* Learning rate is lowered to 3e-5 from the pilot's 1e-4 to reduce forgetting.
* Each candidate is evaluated on the three fixed diagnostic clips and also gets
  the high-frequency diagnostic, because DISTS is the training loss and its own
  number cannot be evidence.

Resumable: an existing final checkpoint is skipped, so an interrupted sweep can
be restarted without retraining finished configurations.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = "/data/disk1/home/ghe270/p3/quality_metrics"
PY = os.path.join(ROOT, "venv", "bin", "python")
TRAIN = os.path.join(ROOT, "fine_tune", "scripts", "train.py")
EVAL = os.path.join(ROOT, "fine_tune", "scripts", "evaluate.py")
HFD = os.path.join(ROOT, "fine_tune", "scripts", "hf_diagnostic.py")
LOGS = os.path.join(ROOT, "fine_tune", "logs")
RESULTS = os.path.join(ROOT, "fine_tune", "results")
CLIPS = ["000", "002", "009"]

LR = 3e-5
STEPS = 900
WEIGHTS = [
    ("ctl_lr3e5_d000", 0.00),
    ("dist_lr3e5_d003", 0.03),
    ("dist_lr3e5_d010", 0.10),
    ("dist_lr3e5_d030", 0.30),
]


def run(cmd, log):
    with open(os.path.join(LOGS, log), "w", encoding="utf-8") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def ckpt_path(name):
    return os.path.join(ROOT, "fine_tune", "runs", name, "step_%06d.pth" % STEPS)


def main():
    sweep = {"lr": LR, "steps": STEPS, "clips": CLIPS,
             "batch": 8, "crop": 64, "reference": "ctl_lr3e5_d000",
             "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "configs": {}}

    for name, dw in WEIGHTS:
        entry = {"dists_weight": dw}
        ck = ckpt_path(name)
        if not os.path.exists(ck):
            t0 = time.time()
            rc = run([PY, TRAIN, "--name", name, "--iterations", str(STEPS),
                      "--batch", "8", "--crop", "64", "--workers", "4",
                      "--dataset-length", "4000", "--lr", str(LR),
                      "--dists-weight", str(dw),
                      "--log-every", "100", "--save-every", str(STEPS)],
                     "sweep_train_%s.log" % name)
            entry["train_rc"] = rc
            entry["train_min"] = round((time.time() - t0) / 60, 1)
            if rc != 0:
                sweep["configs"][name] = entry
                _save(sweep)
                continue
        else:
            entry["train_reused"] = True

        t0 = time.time()
        rc = run([PY, EVAL, "--checkpoint", ck, "--name", name,
                  "--clips", *CLIPS], "sweep_eval_%s.log" % name)
        entry["eval_rc"] = rc
        entry["eval_min"] = round((time.time() - t0) / 60, 1)
        if rc == 0:
            m = json.load(open(os.path.join(RESULTS, name, "metrics.json")))
            entry["mean"] = m["mean"]
            entry["per_clip"] = {c: {k: v[k] for k in
                                     ("psnr", "lpips", "dists", "tlp")}
                                 for c, v in m["clips"].items()}
        sweep["configs"][name] = entry
        _save(sweep)
        print(name, json.dumps(entry.get("mean", {})), flush=True)

    # high-frequency diagnostic for every trained candidate, one process
    names = [n for n in sweep["configs"]
             if os.path.exists(ckpt_path(n))]
    if names:
        args = [PY, HFD, "--checkpoints", "baseline=baseline"]
        args += ["%s=%s" % (n, ckpt_path(n)) for n in names]
        args += ["--clips", "000", "002", "--frames", "10", "40", "70", "95"]
        rc = run(args, "sweep_hf.log")
        sweep["hf_rc"] = rc
        hp = os.path.join(RESULTS, "hf_diagnostic.json")
        if rc == 0 and os.path.exists(hp):
            sweep["hf"] = json.load(open(hp))["results"]
    _save(sweep)
    print("sweep finished", flush=True)


def _save(obj):
    with open(os.path.join(RESULTS, "perceptual_sweep.json"), "w",
              encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
