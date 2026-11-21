import os
os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")
import sys, types, time
import torch
import torchvision
_m = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as F
for n in dir(F):
    setattr(_m, n, getattr(F, n))
sys.modules['torchvision.transforms.functional_tensor'] = _m

from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus

t0 = time.time()
ck = torch.load('/data/disk2/home/ghe270/p3/ckpt/bvsrpp_reds4.pth', map_location='cpu', weights_only=False)
print('ckpt keys', list(ck.keys()), flush=True)
sd = ck['state_dict']
new = {(k[len('generator.'):] if k.startswith('generator.') else k): v
       for k, v in sd.items() if k not in ('step_counter',)}
print('n tensors', len(new), flush=True)

m = BasicVSRPlusPlus(mid_channels=64, num_blocks=7, is_low_res_input=True, spynet_path=None)
miss, unexp = m.load_state_dict(new, strict=False)
print('missing', len(miss), miss[:5], flush=True)
print('unexpected', len(unexp), unexp[:5], flush=True)
print('params M', round(sum(p.numel() for p in m.parameters()) / 1e6, 2), flush=True)

m = m.cuda().eval()
x = torch.rand(1, 5, 3, 180, 320, device='cuda')
with torch.no_grad():
    y = m(x)
print('out', tuple(y.shape), flush=True)
print('elapsed %.1fs' % (time.time() - t0), flush=True)
