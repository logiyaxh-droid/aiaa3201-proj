"""torchvision-backed stand-in for BasicSR's compiled `deform_conv_ext`.

Why this exists
---------------
BasicSR's DeformConvFunction calls into `deform_conv_ext`, a CUDA extension that
must be compiled with the local nvcc. On this machine there is no CUDA toolkit,
and the GPU is an RTX 5070 (sm_120), so compiling is not an option.

`torchvision.ops.deform_conv2d` ships a prebuilt deformable-conv kernel that
supports sm_120, so we forward to it instead. The two must agree on:

  * memory layout -- the C++ kernel indexes offsets as
        offset[b, g, 2*(i*kW + j) + {0,1}, h, w]     (interleaved y,x per tap)
    and torchvision's documented layout is
        [batch, 2 * offset_groups * kH * kW, out_h, out_w]
    with the same (y,x)-per-tap ordering, so the tensors can be passed straight
    through. This is not assumed -- see verify_edvr_local.py, which compares the
    full EDVR output against the server's reference run.
  * padding convention -- BasicSR pads with zeros *and* samples outside the
    feature map as zero; torchvision matches with padding=<same>.
  * im2col_step -- purely a C++ batching optimisation, irrelevant here.

Only inference is implemented; training would need input/offset gradients, which
BasicSR's eval path never asks for.
"""
import torch
import torch.nn.functional as F
import torchvision


def deform_conv_forward(input, weight, offset, output, columns, ones,
                        kW, kH, dW, dH, padW, padH, dilationW, dilationH,
                        group, deformable_group, im2col_step):
    """Mirror of the C++ signature (note the dW/dH-before-pad argument order)."""
    # BasicSR passes stride as (dW, dH) == (width, height); torchvision wants (h, w).
    stride = (dH, dW)
    padding = (padH, padW)
    dilation = (dilationH, dilationW)

    y = torchvision.ops.deform_conv2d(
        input, offset, weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    # The C++ function writes into a preallocated buffer supplied by the caller.
    if output.shape != y.shape:
        raise RuntimeError(
            f"shape mismatch: caller expected {tuple(output.shape)}, "
            f"torchvision produced {tuple(y.shape)}")
    output.copy_(y)
    return output


def deform_conv_backward_input(*args, **kwargs):
    raise NotImplementedError(
        "inference shim only -- BasicSR's eval path does not call this")


def deform_conv_backward_parameters(*args, **kwargs):
    raise NotImplementedError(
        "inference shim only -- BasicSR's eval path does not call this")
