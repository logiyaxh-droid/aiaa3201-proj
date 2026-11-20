# -*- coding: utf-8 -*-
"""Adversarial-weight sweep, run only after a perceptual weight is chosen.

The perceptual term is held FIXED at the chosen value (passed on the command
line) and only the adversarial weight moves, so the two effects stay separable.
A pixel-only control from the perceptual sweep is reused as the reference point.

Adversarial training changes what the numbers mean, so this script records the
generator/discriminator loss traces and refuses to report a run whose
discriminator collapsed (D loss pinned near 0 for a long stretch), because the
adversarial gradient is then zero and the "no change" result would be
uninformative rather than negative.
"""
from __future__ import annotations

import argparse
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


def ckpt(name, steps):
    return os.path.join(ROOT, "fine_tune", "runs", name, "step_%06d.pth" % steps)


def run(cmd, log):
    with open(os.path.join(LOGS, log), "w", encoding="utf-8") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def d_collapsed(train_log, tail=40):
    """True when the discriminator loss sits at ~0 for the recent tail."""
    vals = []
    try:
        for line in open(train_log, encoding="utf-8"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("adv_d") is not None:
                vals.append(row["adv_d"])
    except FileNotFoundError:
        return None
    if len(vals) < 5:
        return None
    tailv = vals[-min(tail, len(vals)):]
    return bool(max(tailv) < 0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dists-weight", type=float, required=True)
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.001, 0.01, 0.05])
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-d", type=float, default=1e-4)
    ap.add_argument("--reference", default="ctl_lr3e5_d000")
    args = ap.parse_args()

    tag = ("d%03d" % round(args.dists_weight * 100))
    sweep = {"dists_weight": args.dists_weight, "steps": args.steps,
             "lr": args.lr, "lr_d": args.lr_d, "clips": CLIPS,
             "reference": args.reference,
             "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "configs": {}}
    path = os.path.join(RESULTS, "adversarial_sweep.json")

    for aw in args.weights:
        name = "adv_lr3e5_%s_a%03d" % (tag, round(aw * 1000))
        entry = {"adv_weight": aw}
        ck = ckpt(name, args.steps)
        if not os.path.exists(ck):
            t0 = time.time()
            rc = run([PY, TRAIN, "--name", name, "--iterations", str(args.steps),
                      "--batch", "8", "--crop", "64", "--workers", "4",
                      "--dataset-length", "4000", "--lr", str(args.lr),
                      "--lr-d", str(args.lr_d),
                      "--dists-weight", str(args.dists_weight),
                      "--adv-weight", str(aw),
                      "--log-every", "50", "--save-every", str(args.steps)],
                     "adv_train_%s.log" % name)
            entry["train_rc"] = rc
            entry["train_min"] = round((time.time() - t0) / 60, 1)
            collapsed = d_collapsed(os.path.join(LOGS, "adv_train_%s.log" % name))
            entry["discriminator_collapsed"] = collapsed
            if rc != 0:
                sweep["configs"][name] = entry
                _save(sweep, path)
                continue
        else:
            entry["train_reused"] = True

        t0 = time.time()
        rc = run([PY, EVAL, "--checkpoint", ck, "--name", name,
                  "--clips", *CLIPS], "adv_eval_%s.log" % name)
        entry["eval_rc"] = rc
        entry["eval_min"] = round((time.time() - t0) / 60, 1)
        if rc == 0:
            m = json.load(open(os.path.join(RESULTS, name, "metrics.json")))
            entry["mean"] = m["mean"]
            entry["per_clip"] = {c: {k: v[k] for k in
                                     ("psnr", "lpips", "dists", "tlp")}
                                 for c, v in m["clips"].items()}
        sweep["configs"][name] = entry
        _save(sweep, path)
        print(name, json.dumps(entry.get("mean", {})), flush=True)

    names = [n for n in sweep["configs"] if os.path.exists(ckpt(n, args.steps))]
    if names:
        cmd = [PY, HFD, "--checkpoints", "baseline=baseline"]
        cmd += ["%s=%s" % (n, ckpt(n, args.steps)) for n in names]
        cmd += ["--clips", "000", "002", "--frames", "10", "40", "70", "95"]
        rc = run(cmd, "adv_hf.log")
        sweep["hf_rc"] = rc
        hp = os.path.join(RESULTS, "hf_diagnostic.json")
        if rc == 0 and os.path.exists(hp):
            sweep["hf"] = json.load(open(hp))["results"]
    _save(sweep, path)
    print("adversarial sweep finished", flush=True)


def _save(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
