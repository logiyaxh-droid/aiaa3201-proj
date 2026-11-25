# -*- coding: utf-8 -*-
p = '/data/disk1/home/ghe270/p3/m2_stageA/scripts/eval_stageA.py'
s = open(p, encoding='utf-8').read()

old = '''        if k > 0:
            lr_t, lr_prev = t_lq[k:k + 1], t_lq[k - 1:k + 1][:1]
            b_prev = full[k - 1:k]
        else:
            lr_t = t_lq[0:1]
            lr_prev = t_lq[0:1]
            b_prev = b_t'''
new = '''        if k > 0:
            lr_t, lr_prev = t_lq[k:k + 1], t_lq[k - 1:k + 1][:1]
            b_prev = full[k - 1:k]
        elif len(ids) > 1:
            # First frame: use the NEXT frame as the neighbour instead of
            # b_t itself.  With b_prev = b_t the difference is identically 0,
            # u_ref clamps to 1e-6 and u1 saturates to 1.000 across the whole
            # first frame -- a strong, silent gate error on 1 of every 9 frames.
            lr_t, lr_prev = t_lq[0:1], t_lq[1:2]
            b_prev = full[1:2]
        else:
            lr_t, lr_prev, b_prev = t_lq[0:1], t_lq[0:1], b_t'''
assert old in s
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print("eval_stageA.py patched (fix 7)")
