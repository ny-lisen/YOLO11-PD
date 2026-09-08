import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Union

from ..modules.block import C3k
from ..modules.conv import Conv, DWConv, RepConv, GhostConv, autopad
from ..modules.block import *
from .wtconv2d import *
try:
    from mmcv.cnn import build_activation_layer, build_norm_layer
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
    from mmengine.model import constant_init, normal_init
except ImportError as e:
    pass



__all__ = ['WTConv2d', 'C2f_WTConv', 'C3k2_WTConv', 'SCSP', 'Fusion', 'CSP_MSCB', 'EUCB']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


##-- C2f_WTConv
class Bottleneck_WTConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        # self.cv1 = WTConv2d(c1, c2)
        self.cv2 = WTConv2d(c2, c2)

class C2f_WTConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_WTConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class C3k_WTConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_WTConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_WTConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_WTConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_WTConv(self.c, self.c, shortcut, g) for _ in range(n))




##--

##-- Shared Conv Spatial Pyramid，SCSP
class SCSP(nn.Module):
    def __init__(self, c1, c2, dilations=[1, 3, 5]) -> None:
        super().__init__()

        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (1 + len(dilations)), c2, 1, 1)
        self.share_conv = nn.Conv2d(in_channels=c_, out_channels=c_, kernel_size=3, stride=1, padding=1, bias=False)
        self.dilations = dilations

    def forward(self, x):
        y = [self.cv1(x)]
        for dilation in self.dilations:
            y.append(F.conv2d(y[-1], weight=self.share_conv.weight, bias=None, dilation=dilation,
                              padding=(dilation * (3 - 1) + 1) // 2))
        return self.cv2(torch.cat(y, 1))
##--

##-- DyDCNv2 for DRDetect
class DyDCNv2(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.
    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset, mask)
        if self.with_norm:
            x = self.norm(x)
        return x
##--

##-- Fusion: there are four ways, and we choose 'bifpn'
class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()

        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn']
        self.fusion = fusion

        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])
            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)

    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
##--

##-- Multi-scale depth-wise convolution (MSDC)
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                Conv(self.in_channels, self.in_channels, kernel_size, s=stride, g=self.in_channels)
            )
            for kernel_size in self.kernel_sizes
        ])

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        return outputs


class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB)
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5], stride=1, expansion_factor=2,
                 dw_parallel=True, add=True):
        super(MSCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            Conv(self.in_channels, self.ex_channels, 1)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            Conv(self.combined_channels, self.out_channels, 1, act=False)
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = self.channel_shuffle(dout, math.gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class CSP_MSCB(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=[1, 3, 5], shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(MSCB(self.c, self.c, kernel_sizes=kernel_sizes) for _ in range(n))
##--

##--   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels, s=stride)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x
##--
