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

import torch
from torch import nn


class ROIsDistributorFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, rois, levels_num, canonical_level=4, canonical_scale=224):
        return g.op('ExperimentalDetectronROIsRedistributor', rois, levels_num_i=levels_num, canonical_level_i=canonical_level,
                    canonical_scale_i=canonical_scale, outputs=levels_num)

    @staticmethod
    def forward(ctx, rois, levels_num, canonical_level=4, canonical_scale=224):
        areas = ((rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1)).view(-1)
        areas.sqrt_()
        areas /= canonical_scale
        areas += 1e-6
        areas.log2_()
        target_levels = torch.floor(areas + canonical_level)
        target_levels.clamp_(min=0, max=levels_num - 1)
        level_indices = tuple((target_levels == i).nonzero().view(-1) for i in range(levels_num))
        return level_indices


def redistribute_rois(rois, levels_num, canonical_level=4, canonical_scale=224):
    return ROIsDistributorFunction.apply(rois, levels_num, canonical_level, canonical_scale)


class ROIsDistributor(nn.Module):
    def __init__(self, levels_num, canonical_level=2, canonical_scale=224):
        super().__init__()
        self.levels_num = levels_num
        self.canonical_level = canonical_level
        self.canonical_scale = canonical_scale

    def forward(self, rois, scores):
        return redistribute_rois(rois, self.levels_num, self.canonical_level, self.canonical_scale)
