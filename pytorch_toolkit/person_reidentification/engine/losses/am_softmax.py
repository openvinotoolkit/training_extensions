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

from __future__ import absolute_import
from __future__ import division

import math

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import Parameter


class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""

    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class AMSoftmaxLoss(nn.Module):
    margin_types = ['cos', 'arc']

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True,
                 conf_penalty=0.,
                 margin_type='cos', gamma=0., m=0.5, s=30, t=1.):
        super(AMSoftmaxLoss, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.conf_penalty = conf_penalty
        self.logsoftmax = nn.LogSoftmax(dim=1)

        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m >= 0
        self.m = m
        assert s > 0
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        assert t >= 1
        self.t = t

    def get_last_info(self):
        return {}

    def forward(self, cos_theta, target):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """

        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.m
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m #cos(theta+m)
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        if self.gamma == 0 and self.t == 1.:
            if self.conf_penalty > 0.:
                output *= self.s
                log_probs = self.logsoftmax(output)
                probs = torch.exp(log_probs)
                ent = (-probs*torch.log(probs.clamp(min=1e-12))).sum(1)
                loss = F.relu(F.cross_entropy(output, target, reduction='none') - self.conf_penalty * ent)
                with torch.no_grad():
                    nonzero_count = loss.nonzero().size(0)
                return loss.sum() / nonzero_count
            else:
                return F.cross_entropy(self.s*output, target)

        if self.t > 1:
            h_theta = self.t - 1 + self.t*cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)
            return F.cross_entropy(self.s*output, target)

        return focal_loss(F.cross_entropy(self.s*output, target, reduction='none'), self.gamma)
