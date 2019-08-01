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

from __future__ import division

from itertools import product
from math import sqrt

import torch
from torch import nn


class PriorBox(nn.Module):
    def __init__(self, min_size, max_size, aspect_ratio, flip, clip, variance, step, offset,
                 step_h, step_w, img_size, img_h, img_w):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratio = aspect_ratio
        self.flip = flip
        self.clip = clip
        self.variance = variance
        self.step = step
        self.offset = offset
        self.step_h = step_h
        self.step_w = step_w
        self.img_size = img_size
        self.img_h = img_h
        self.img_w = img_w

    def forward(self, input_fm, img_tensor):
        return PriorBoxFunction.apply(input_fm, img_tensor, self)


class PriorBoxFunction(torch.autograd.Function):
    """Compute priorbox coordinates in point form for each source
    feature map.
    """

    @staticmethod
    def symbolic(g, input_fm, img_tensor, priorbox_params):
        return g.op("PriorBox", input_fm, img_tensor, min_size_f=[priorbox_params.min_size],
                    max_size_f=[priorbox_params.max_size],
                    aspect_ratio_f=priorbox_params.aspect_ratio, flip_i=priorbox_params.flip,
                    clip_i=priorbox_params.clip,
                    variance_f=priorbox_params.variance, step_f=priorbox_params.step,
                    offset_f=priorbox_params.offset, step_h_f=priorbox_params.step_h, step_w_f=priorbox_params.step_w,
                    img_size_i=priorbox_params.img_size, img_h_i=priorbox_params.img_h, img_w_i=priorbox_params.img_w)

    @staticmethod
    def forward(ctx, input_fm, img_tensor, priorbox_params):
        for v in priorbox_params.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

        mean = []
        variance_channel = []
        f_h = input_fm.size()[2]
        f_w = input_fm.size()[3]
        img_height = img_tensor.size()[2]
        img_width = img_tensor.size()[3]

        box_widths_heights = [(priorbox_params.min_size, priorbox_params.min_size),
                              (sqrt(priorbox_params.min_size * priorbox_params.max_size),
                               sqrt(priorbox_params.min_size * priorbox_params.max_size))]
        for ar in priorbox_params.aspect_ratio:
            box_widths_heights.append((priorbox_params.min_size * sqrt(ar), priorbox_params.min_size / sqrt(ar)))
            if priorbox_params.flip:
                box_widths_heights.append((priorbox_params.min_size / sqrt(ar), priorbox_params.min_size * sqrt(ar)))

        for i, j in product(range(f_h), range(f_w)):
            # unit center x,y
            cx = (j + priorbox_params.offset) * priorbox_params.step
            cy = (i + priorbox_params.offset) * priorbox_params.step

            for box_width, box_height in box_widths_heights:
                mean += [(cx - box_width / 2.) / img_width, (cy - box_height / 2.) / img_height,
                         (cx + box_width / 2.) / img_width, (cy + box_height / 2.) / img_height]
                variance_channel += priorbox_params.variance

        # back to torch land
        mean = torch.Tensor(mean).unsqueeze(0)
        if priorbox_params.clip:
            mean.clamp_(max=1, min=0)
        variance_channel = torch.Tensor(variance_channel).unsqueeze(0)
        output = torch.stack((mean, variance_channel), dim=1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
