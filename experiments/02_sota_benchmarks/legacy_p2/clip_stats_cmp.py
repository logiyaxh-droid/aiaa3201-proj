import json
s = json.load(open("/tmp/p2/clip_stats.json"))
e = json.load(open("/tmp/p2/metrics_edvr.json"))["per_clip"]
t = json.load(open("/tmp/p2/metrics_tecogan.json"))["per_clip"]
print("%-5s %7s %7s %8s %8s %8s" % ("clip", "sharp", "motion", "edvr", "teco", "gap"))
for c, v in sorted(s.items()):
    print("%-5s %7.1f %7.2f %8.2f %8.2f %8.2f" % (c, v["sharp"], v["motion"], e[c]["psnr"], t[c]["psnr"], e[c]["psnr"] - t[c]["psnr"]))
