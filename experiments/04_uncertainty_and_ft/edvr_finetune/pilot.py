# -*- coding: utf-8 -*-
"""Pilot: does the perceptual objective actually move DISTS at a usable learning rate?

Runs two short trainings on the same seed and schedule:
  A. pixel only            (control -- isolates what plain fine-tuning does)
  B. pixel + DISTS 0.1     (the candidate)

Each is evaluated on clip 000 only, at two checkpoints, so the sign of the change
and its PSNR cost are visible before committing to the full sweep.  This is a
directional check, not a result.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = "/data/disk1/home/ghe270/p3/quality_metrics"
PY = os.path.join(ROOT, "venv", "bin", "python")
TRAIN = os.path.join(ROOT, "fine_tune", "scripts", "train.py")
EVAL = os.path.join(ROOT, "fine_tune", "scripts", "evaluate.py")
LOGS = os.path.join(ROOT, "fine_tune", "logs")
os.makedirs(LOGS, exist_ok=True)

PILOT_STEPS = 400
CONFIGS = [
    ("pilot_control", 0.0),
    ("pilot_dists010", 0.1),
]


def run(cmd, log_name):
    path = os.path.join(LOGS, log_name)
    with open(path, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode, path


def main():
    report = {}
    for name, dw in CONFIGS:
        rc, _ = run([PY, TRAIN, "--name", name, "--iterations", str(PILOT_STEPS),
                     "--batch", "8", "--crop", "64", "--workers", "4",
                     "--dataset-length", "2000", "--lr", "1e-4",
                     "--dists-weight", str(dw),
                     "--log-every", "50", "--save-every", str(PILOT_STEPS)],
                    "pilot_train_%s.log" % name)
        if rc != 0:
            print("train failed for %s" % name, flush=True)
            report[name] = {"train_rc": rc}
            continue
        ckpt = os.path.join(ROOT, "fine_tune", "runs", name,
                            "step_%06d.pth" % PILOT_STEPS)
        rc2, _ = run([PY, EVAL, "--checkpoint", ckpt, "--name", name,
                      "--clips", "000"], "pilot_eval_%s.log" % name)
        m = json.load(open(os.path.join(ROOT, "fine_tune", "results", name,
                                        "metrics.json")))
        row = m["clips"]["000"]
        report[name] = {"dists_weight": dw, "step": PILOT_STEPS,
                        "psnr": row["psnr"], "lpips": row["lpips"],
                        "dists": row["dists"], "tlp": row["tlp"]}
        print(name, json.dumps(report[name]), flush=True)

    base = json.load(open(os.path.join(ROOT, "fine_tune", "results", "baseline",
                                       "metrics.json")))["clips"]["000"]
    report["baseline"] = {k: base[k] for k in ("psnr", "lpips", "dists", "tlp")}
    print("baseline", json.dumps(report["baseline"]), flush=True)

    with open(os.path.join(ROOT, "fine_tune", "results", "pilot.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
