# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import torch
import torch.nn as nn

# pylint: disable=W0212
class LossWrapper(nn.modules.loss._Loss):
    def __init__(self, loss, input_index, target_index, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(LossWrapper, self).__init__(size_average, reduce, reduction)
        self.loss = loss
        self.target_index = target_index
        self.input_index = input_index

    # pylint: disable=W0221
    def forward(self, input, target):
        return self.loss(input[self.input_index], target[self.target_index])

class Dice_loss_joint(nn.Module):
    def __init__(self, index=0, priority=1):
        super(Dice_loss_joint, self).__init__()
        self.index = index
        self.priority = priority

    # pylint: disable=W0221
    def forward(self, x, y):
        assert x[self.index].shape == y[self.index].shape
        N, C, _, _, _ = x[self.index].shape

        pred = x[self.index].view(N, C, -1)
        gt = y[self.index].view(N, C, -1)

        intersection = (pred*gt).sum(dim=(0, 2)) + 1e-6
        union = (pred**2 + gt).sum(dim=(0, 2)) + 2e-6

        dice = 2.0*intersection / union

        return self.priority*(1.0 - torch.mean(dice))
