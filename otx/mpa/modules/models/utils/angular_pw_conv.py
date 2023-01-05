# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalize import normalize


class AngularPWConv(nn.Module):
    def __init__(self, in_features, out_features, clip_output=False):
        super(AngularPWConv, self).__init__()

        self.in_features = in_features
        assert in_features > 0
        self.out_features = out_features
        assert out_features >= 2
        self.clip_output = clip_output

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.normal_().renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        weight = normalize(self.weight, dim=1, p=2).view(self.out_features, self.in_features, 1, 1)
        out = F.conv2d(x, weight)

        if self.clip_output and not torch.onnx.is_in_onnx_export():
            out = out.clamp(-1.0, 1.0)

        return out
