# -*- coding: utf-8 -*-
"""M2-B dev-set Go/No-Go judge (guide section 5.1)."""
import json

import numpy as np

DEV = ['000', '002', '009']
BASE = '/data/disk1/home/ghe270/p3/prior_qualification/results/reds_real_c9/rbv/metrics.json'
OUT = '/data/disk1/home/ghe270/p3/m2b_rbv_ft/results'


def mean_of(cm):
    out = {k: float(np.mean([cm[c][k] for c in DEV]))
           for k in ('psnr', 'ssim', 'lpips', 'dists', 'tlp')}
    out['e_warp'] = float(np.mean([cm[c]['temporal']['e_warp_sr'] for c in DEV]))
    return out


def main():
    b = mean_of(json.load(open(BASE))['clip_means'])
    print("RealBasicVSR baseline on dev 000/002/009 (27 frames)")
    print("  PSNR %.4f  LPIPS %.5f  DISTS %.5f  tLP %.5f  E_warp %.4f"
          % (b['psnr'], b['lpips'], b['dists'], b['tlp'], b['e_warp']))
    print("\nthresholds: LPIPS <= %.5f (-7%%), PSNR >= %.4f (-0.42 dB), "
          "E_warp <= %.4f" % (b['lpips'] * 0.93, b['psnr'] - 0.42, 5.15))

    for arm in ("b1_dev", "b2_dev", "b2b_dev"):
        try:
            a = mean_of(json.load(open('%s/%s/metrics.json' % (OUT, arm)))
                        ['clip_means'])
        except FileNotFoundError:
            print("\n%s: no metrics" % arm)
            continue
        lp = (b['lpips'] - a['lpips']) / b['lpips']
        drop = b['psnr'] - a['psnr']
        er = a['e_warp']
        ok_lp = a['lpips'] <= b['lpips'] * 0.93
        ok_ps = a['psnr'] >= b['psnr'] - 0.42
        ok_ew = er <= 5.15
        print("\n[%s]" % arm)
        print("  PSNR  %.4f (%+.4f dB)   %s"
              % (a['psnr'], -drop, "PASS" if ok_ps else "FAIL"))
        print("  LPIPS %.5f (%+.2f%%)   %s"
              % (a['lpips'], lp * 100, "PASS" if ok_lp else "FAIL"))
        print("  DISTS %.5f (%+.1f%%)"
              % (a['dists'], (a['dists'] - b['dists']) / b['dists'] * 100))
        print("  E_warp %.4f (%+.2f%%)  %s"
              % (er, (er - b['e_warp']) / b['e_warp'] * 100,
                 "PASS" if ok_ew else "FAIL"))
        print("  tLP   %.5f (%+.1f%%)"
              % (a['tlp'], (a['tlp'] - b['tlp']) / b['tlp'] * 100))
        print("  -> %s" % ("GO" if (ok_lp and ok_ps and ok_ew) else "NO-GO"))


if __name__ == "__main__":
    main()
