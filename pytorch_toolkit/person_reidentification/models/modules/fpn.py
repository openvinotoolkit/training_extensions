"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import math
import operator
from functools import reduce

from torch import nn, Tensor
from torch.nn import init
import torch.nn.functional as F


def xavier_fill(tensor):
    """Caffe2 XavierFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_in = size / tensor.shape[0]
    scale = math.sqrt(3 / fan_in)
    return init.uniform_(tensor, -scale, scale)


class FeatureExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dims_in = ()
        self.scales_in = ()
        self.dims_out = ()
        self.scales_out = ()

    def forward(self, *inputs):
        raise NotImplementedError


# FIXME. Move it to a proper place.
def duplicate(x, n, copy=False):
    if copy:
        return list([x for _ in range(n)])
    else:
        return [x, ] * n


class TopDownLateral(nn.Module):
    def __init__(self, dim_in_top, dim_in_lateral, dim_out, zero_init_lateral=False, group_norm=False):
        super().__init__()
        self.dim_in_top = dim_in_top
        self.dim_in_lateral = dim_in_lateral
        self.dim_out = dim_out
        self.zero_init_lateral = zero_init_lateral
        self.group_norm = group_norm

        self.conv_lateral = nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0)

        self._init_weights()

    def forward(self, top_blob, lateral_blob):
        # Lateral 1x1 conv
        lat = self.conv_lateral(lateral_blob)
        # Top-down 2x upsampling
        if lat.shape != top_blob.shape:
            td = F.interpolate(top_blob, scale_factor=2, mode='nearest')
        else:
            td = top_blob
        # Sum lateral and top-down
        lat += td
        return lat

    def _init_weights(self):
        if self.group_norm:
            conv = self.conv_lateral[0]
        else:
            conv = self.conv_lateral

        if self.zero_init_lateral:
            nn.init.constant_(conv.weight, 0)
        else:
            xavier_fill(conv.weight)
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)


class FPN(FeatureExtractor):
    def __init__(self, dims_in, scales_in, dims_internal, dims_out, topdown_lateral_block=TopDownLateral,
                 group_norm=False, **kwargs):
        super().__init__(**kwargs)

        self.dims_in = dims_in
        self.scales_in = scales_in
        self.dims_out = dims_out
        self.scales_out = list(scales_in)
        self.scales_out.append(self.scales_out[-1] * 2)
        self.group_norm = group_norm

        if not isinstance(dims_in, (tuple, list)):
            dims_in = (dims_in, )
        n = len(dims_in)
        if not isinstance(dims_internal, (tuple, list)):
            dims_internal = duplicate(dims_internal, len(dims_in))
            self.dims_internal = dims_internal
        if not isinstance(dims_out, (tuple, list)):
            dims_out = duplicate(dims_out, len(dims_in))
        assert len(dims_in) == len(dims_internal) == len(dims_out)
        self.dims_out = dims_out

        self.conv_top = nn.Conv2d(dims_in[-1], dims_internal[0], 1, 1, 0)

        self.topdown_lateral = nn.ModuleList()
        self.posthoc = nn.ModuleList()
        # Add top-down and lateral connections
        for dim_in, dim_inside in zip(reversed(dims_in[:-1]), reversed(dims_internal[:-1])):
            self.topdown_lateral.append(topdown_lateral_block(dim_inside, dim_in, dim_inside, group_norm=self.group_norm))
        # Post-hoc scale-specific 3x3 convs
        for dim_inside, dim_out in zip(dims_internal, dims_out):
            self.posthoc.append(nn.Conv2d(dim_inside, dim_out, 3, 1, 1))

        self.extra_maxpool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
        self.init_weights()

    def init_weights(self):
        for m in (self.conv_top, *self.posthoc):
            if isinstance(m, nn.Conv2d):
                xavier_fill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if isinstance(x, Tensor):
            x = [x, ]
        assert len(self.dims_in) == len(x)

        top_blob = self.conv_top(x[-1])
        fpn_output_blobs = [self.posthoc[0](top_blob)]
        for lateral_blob, topdown_lateral_module, posthoc_module in \
                zip(reversed(x[:-1]), self.topdown_lateral, self.posthoc[1:]):
            top_blob = topdown_lateral_module(top_blob, lateral_blob)
            fpn_output_blobs.append(posthoc_module(top_blob))
        fpn_output_blobs = [self.extra_maxpool(fpn_output_blobs[0]), ] + fpn_output_blobs

        return list(reversed(fpn_output_blobs))
