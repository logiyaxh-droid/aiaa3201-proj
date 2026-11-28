import hashlib
import json
import os

ROOT = "/data/disk1/shared/aiaa3201/project1_video_sr/val_sharp"
CLIPS = ["000", "011", "015", "020"]
IDS = [0, 49, 98]
OUT = "/data/disk2/home/ghe270/p0/out/audit/course_val_sharp_hashes.json"

out = {}
for c in CLIPS:
    out[c] = {}
    for i in IDS:
        p = os.path.join(ROOT, c, "%08d.png" % i)
        with open(p, "rb") as f:
            b = f.read()
        out[c][i] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=1))
