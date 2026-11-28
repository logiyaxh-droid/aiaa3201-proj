# -*- coding: utf-8 -*-
import json

import numpy as np

DEV = ['000', '002', '009']
BASE = '/data/disk1/home/ghe270/p3/prior_qualification/results/reds_real_c9/rbv/metrics.json'
P4 = '/data/disk1/home/ghe270/p3/m2_stageA/results/pilot_p4/metrics.json'


def mean_of(cm, keys):
    out = {k: float(np.mean([cm[c][k] for c in DEV])) for k in keys}
    out['e_warp'] = float(np.mean([cm[c]['temporal']['e_warp_sr'] for c in DEV]))
    return out


def main():
    base = mean_of(json.load(open(BASE))['clip_means'],
                   ['psnr', 'ssim', 'lpips', 'dists', 'tlp'])
    a = mean_of(json.load(open(P4))['clip_means'],
                ['psnr', 'ssim', 'lpips', 'dists', 'tlp'])
    print("P4_perc_fix on dev clips 000/002/009 (27 frames)")
    print("%-8s %10s %10s %10s" % ("metric", "RBV", "P4", "delta"))
    for k in ['psnr', 'ssim', 'lpips', 'dists', 'tlp']:
        print("%-8s %10.4f %10.4f %+10.4f" % (k, base[k], a[k], a[k] - base[k]))
    print("%-8s %10.4f %10.4f %+10.4f"
          % ('e_warp', base['e_warp'], a['e_warp'], a['e_warp'] - base['e_warp']))

    lp_imp = (base['lpips'] - a['lpips']) / base['lpips']
    drop = base['psnr'] - a['psnr']
    er = a['e_warp'] / base['e_warp']
    print()
    print("LPIPS improvement: %+.2f%%  (need >= +5%%)   %s"
          % (lp_imp * 100, "PASS" if lp_imp >= 0.05 else "FAIL"))
    print("PSNR drop        : %.4f dB (need <= 0.40)  %s"
          % (drop, "PASS" if drop <= 0.40 else "FAIL"))
    print("E_warp ratio     : %.3f    (need <= 1.15)  %s"
          % (er, "PASS" if er <= 1.15 else "FAIL"))


if __name__ == "__main__":
    main()
