"""BasicSR imports torchvision.transforms.functional_tensor, removed in tv>=0.17.
Install a shim before importing basicsr. Usage:  import compat_shim  (first)
"""
import sys, types
import torchvision
m = types.ModuleType('torchvision.transforms.functional_tensor')
from torchvision.transforms import functional as _F
for _n in dir(_F):
    if not _n.startswith('__'):
        setattr(m, _n, getattr(_F, _n))
sys.modules['torchvision.transforms.functional_tensor'] = m
