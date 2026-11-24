p = "scripts/score_realvsr.py"
s = open(p).read()

if "RVQ_DEV" in s:
    print("already patched")
else:
    a = '    lp = EM.LPIPS(device="cuda", torch_home="/tmp/torch_home")'
    b = ('    dev = os.environ.get("RVQ_DEV", "cuda")\n'
         '    lp = EM.LPIPS(device=dev, torch_home="/tmp/torch_home")')
    s = s.replace(a, b)
    s = s.replace('    dists = C.DistsMetric(device="cuda")',
                  '    dists = C.DistsMetric(device=dev)')
    s = s.replace('device="cuda").mean()', 'device=dev).mean()')
    open(p, "w").write(s)
    print("patched")
