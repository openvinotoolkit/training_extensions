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

from math import sqrt

import numpy as np
import torch
from torch import nn


def meshgrid(a, b):
    x = a.repeat(len(b))
    y = b.repeat(len(a), 1).t().contiguous().view(-1)
    return x, y


class PriorGridGenerator(torch.autograd.Function):
    @staticmethod
    def symbolic(g, prior_boxes, feature_map, im_data, w=0, h=0, stride_x=0, stride_y=0, flatten=True):
        return g.op('ExperimentalDetectronPriorGridGenerator', prior_boxes, feature_map, im_data,
                    w_i=int(w), h_i=int(h), stride_x_f=stride_x, stride_y_f=stride_y, flatten_i=int(flatten))

    @staticmethod
    def forward(ctx, prior_boxes, feature_map, im_data, w=0, h=0, stride_x=0, stride_y=0, flatten=True):
        device = prior_boxes.device

        grid_w = w if w else feature_map.shape[-1]
        grid_h = h if h else feature_map.shape[-2]
        s_x = stride_x if stride_x else im_data.shape[-1] / feature_map.shape[-1]
        s_y = stride_y if stride_y else im_data.shape[-2] / feature_map.shape[-2]
        # grid_w = feature_map.shape[-1]
        # grid_h = feature_map.shape[-2]
        # s_x = im_data.shape[-1] / feature_map.shape[-1]
        # s_y = im_data.shape[-2] / feature_map.shape[-2]
        # print(w, grid_w, h, grid_h, stride_x, s_x, stride_y, s_y, feature_map.shape, im_data.shape)
        # grid_w, grid_h, s_x, s_y = w, h, stride_x, stride_y

        shift_x, shift_y = meshgrid(
            torch.linspace(0, grid_w - 1, grid_w, dtype=torch.float32) * s_x + s_x * 0.5,
            torch.linspace(0, grid_h - 1, grid_h, dtype=torch.float32) * s_y + s_y * 0.5)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y)).t().to(device=device, dtype=torch.float32)
        prior_boxes_grid = (prior_boxes.unsqueeze(0) + shifts.unsqueeze(1))
        if flatten:
            prior_boxes_grid = prior_boxes_grid.reshape(-1, 4)
        else:
            prior_boxes_grid = prior_boxes_grid.reshape(grid_h, grid_w, -1, 4)
        return prior_boxes_grid


class PriorBox(nn.Module):
    def __init__(self, widths=None, heights=None, aspect_ratios=None, min_size=None, max_size=None, flatten=True,
                 use_cache=True):
        super().__init__()
        self.flatten = flatten

        assert (widths is not None and heights is not None) != \
               (min_size is not None and max_size is not None and aspect_ratios is not None),\
            'Please specify either prior boxes widths and heights, or their min/max sizes and aspect ratios.'
        self.widths = widths
        self.heights = heights
        if self.widths is None or self.heights is None:
            self.widths = [min_size, sqrt(min_size * max_size)]
            self.heights = [min_size, sqrt(min_size * max_size)]
            for ar in aspect_ratios:
                self.widths.extend([min_size * sqrt(ar), min_size / sqrt(ar)])
                self.heights.extend([min_size / sqrt(ar), min_size * sqrt(ar)])

        self.prior_boxes_numpy = []
        for w, h in zip(self.widths, self.heights):
            self.prior_boxes_numpy.append([-w / 2, -h / 2, w / 2, h / 2])
        self.prior_boxes_numpy = np.array(self.prior_boxes_numpy, dtype=np.float32)

        self.use_cache = use_cache
        self.register_buffer('prior_boxes', torch.from_numpy(np.copy(self.prior_boxes_numpy)))

    @staticmethod
    def meshgrid(a, b):
        x = a.repeat(len(b))
        y = b.repeat(len(a), 1).t().contiguous().view(-1)
        return x, y

    def priors_num(self):
        return self.prior_boxes.size(0)

    def forward(self, feature_map, im_data, w=0, h=0, stride_x=0, stride_y=0):
        if self.use_cache:
            prior_boxes = self.prior_boxes
        else:
            prior_boxes = torch.from_numpy(self.prior_boxes_numpy).to(feature_map.device)
        return PriorGridGenerator.apply(prior_boxes, feature_map, im_data, int(w), int(h),
                                        stride_x, stride_y, self.flatten)
