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

from nncf.dynamic_graph.patch_pytorch import register_operator
from nncf.functions import STThreshold, logit


def binary_mask(mask):
    return STThreshold.apply(torch.sigmoid(mask))


@register_operator()
def calc_rb_binary_mask(mask, uniform_buffer, eps):
    if uniform_buffer is not None:
        uniform_buffer.uniform_()
        mask = mask + logit(uniform_buffer.clamp(eps, 1 - eps))
    return binary_mask(mask)
