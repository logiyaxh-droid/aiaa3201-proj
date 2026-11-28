#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel ranged download: many HTTP connections, one file.

hf-mirror throttles a single connection hard (measured 64 KB/s while a burst
request reached 2 MB/s), so split the file into fixed-size ranges and fetch
them concurrently.  Each worker resumes from its own part file, so a dropped
connection only costs that slice.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import requests

URL = sys.argv[1]
OUT = sys.argv[2]
NTHREAD = int(sys.argv[3]) if len(sys.argv) > 3 else 16
SLICE = 8 * 1024 * 1024
RETRY = 40

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

head = requests.head(URL, allow_redirects=True, timeout=60)
total = int(head.headers.get("x-linked-size") or head.headers.get("Content-Length"))
print("total", total, flush=True)

n_slices = (total + SLICE - 1) // SLICE
parts_dir = OUT + ".parts"
os.makedirs(parts_dir, exist_ok=True)

done = [0] * n_slices
lock = threading.Lock()
stop = False


def part_path(i):
    return os.path.join(parts_dir, "%05d" % i)


def worker(i):
    global stop
    start = i * SLICE
    end = min(total, start + SLICE) - 1
    want = end - start + 1
    pp = part_path(i)
    for attempt in range(RETRY):
        if stop:
            return
        have = os.path.getsize(pp) if os.path.exists(pp) else 0
        if have == want:
            with lock:
                done[i] = 1
            return
        if have > want:
            os.remove(pp)
            have = 0
        headers = {"Range": "bytes=%d-%d" % (start + have, end),
                   "Accept-Encoding": "identity"}
        try:
            with requests.get(URL, headers=headers, stream=True,
                              timeout=(30, 90)) as r:
                if r.status_code not in (206, 200):
                    raise RuntimeError("status %s" % r.status_code)
                mode = "ab" if have else "wb"
                with open(pp, mode) as f:
                    for chunk in r.iter_content(1 << 18):
                        if chunk:
                            f.write(chunk)
        except Exception as e:  # noqa: BLE001
            if attempt % 5 == 0:
                print("slice %d retry %d: %s" % (i, attempt, e), flush=True)
            time.sleep(2 + attempt * 0.5)
    with lock:
        done[i] = 2


threads = [threading.Thread(target=worker, args=(i,), daemon=True)
           for i in range(n_slices)]
for t in threads:
    t.start()

t0 = time.time()
while any(t.is_alive() for t in threads):
    time.sleep(20)
    got = sum(os.path.getsize(part_path(i)) for i in range(n_slices)
              if os.path.exists(part_path(i)))
    el = time.time() - t0
    print("progress %.1f%% %d/%d MB %.0f KB/s"
          % (100.0 * got / total, got // 2**20, total // 2**20,
             got / 1024 / max(el, 1)), flush=True)

failed = [i for i in range(n_slices) if done[i] != 1]
if failed:
    print("FAILED slices", failed[:20], "of", n_slices, flush=True)
    sys.exit(1)

with open(OUT, "wb") as out:
    for i in range(n_slices):
        with open(part_path(i), "rb") as f:
            out.write(f.read())
size = os.path.getsize(OUT)
print("DONE size=%d expected=%d match=%s" % (size, total, size == total), flush=True)
if size != total:
    sys.exit(1)
