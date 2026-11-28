# -*- coding: utf-8 -*-
import json

B = '/data/disk1/home/ghe270/p3/prior_qualification/results/reds_real_c9/rbv/metrics.json'
P = '/data/disk1/home/ghe270/p3/m2_stageA/results/pilot_p4/metrics.json'

b = json.load(open(B))['clip_means']
a = json.load(open(P))['clip_means']
for c in ['000', '002', '009', '016', '005']:
    if c not in b or c not in a:
        continue
    print("clip %s | PSNR %.3f -> %.3f (%+.3f) | LPIPS %.4f -> %.4f (%+.4f) "
          "| E_warp %.2f -> %.2f"
          % (c, b[c]['psnr'], a[c]['psnr'], a[c]['psnr'] - b[c]['psnr'],
             b[c]['lpips'], a[c]['lpips'],
             a[c]['lpips'] - b[c]['lpips'],
             b[c]['temporal']['e_warp_sr'],
             a[c]['temporal']['e_warp_sr']))
