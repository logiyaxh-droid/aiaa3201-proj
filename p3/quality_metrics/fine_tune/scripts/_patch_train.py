# -*- coding: utf-8 -*-
"""Small targeted fixes to the training script, kept explicit for review."""
import io
P = "/data/disk1/home/ghe270/p3/quality_metrics/fine_tune/scripts/train.py"
s = io.open(P, encoding="utf-8").read()

s = s.replace('''def charbonnier(pred, target, eps=1e-3):
    """BasicSR-compatible Charbonnier mean; eps=1e-3 (squared under the root)."""
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()
''', '''def charbonnier(pred, target, eps=1e-3):
    """Exact BasicSR CharbonnierLoss mean reduction (eps is squared)."""
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()
''')

old = '''        lr, gt = lr.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        pred = net(lr)
        pixel = F.charbonnier_loss(pred, gt, reduction="mean") if hasattr(F, "charbonnier_loss") else charbonnier(pred, gt)'''
new = '''        lr, gt = lr.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        pred = net(lr)
        pixel = charbonnier(pred, gt)'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('''            row = {"step": step, "lr": lr.shape[0],''',
              '''            row = {"step": step, "batch_size": lr.shape[0],''', 1)

# Make the single-variable nature of a run machine-checkable.
old2 = '''    out = os.path.join(ROOT, "fine_tune", "runs", args.name)
    os.makedirs(out, exist_ok=True)
    config = vars(args).copy()'''
new2 = '''    if args.lpips_weight and args.dists_weight:
        raise ValueError("perceptual scan changes one perceptual loss at a time")
    if args.lpips_weight < 0 or args.dists_weight < 0 or args.adv_weight < 0:
        raise ValueError("loss weights must be non-negative")
    out = os.path.join(ROOT, "fine_tune", "runs", args.name)
    os.makedirs(out, exist_ok=True)
    config = vars(args).copy()'''
assert old2 in s
s = s.replace(old2, new2, 1)
io.open(P, "w", encoding="utf-8").write(s)
print("fixed variable shadowing and guarded loss configuration")
