"""
 Copyright (c) 2019-2020 Intel Corporation
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


def abs_magnitude(weight):
    return torch.abs(weight)


def normed_magnitude(weight):
    return torch.abs(weight) / weight.norm(2)


WEIGHT_IMPORTANCE_FUNCTIONS = {
    'abs': abs_magnitude,
    'normed_abs': normed_magnitude
}


def calc_magnitude_binary_mask(weight, weight_importance, threshold):
    return (weight_importance(weight) > threshold).float()
