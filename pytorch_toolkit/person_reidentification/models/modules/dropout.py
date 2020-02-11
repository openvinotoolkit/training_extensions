"""
 Copyright (c) 2020 Intel Corporation

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
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Module):
    DISTRIBUTIONS = ['bernoulli', 'gaussian', 'none']

    def __init__(self, p=0.5, mu=0.5, sigma=0.2, dist='bernoulli'):
        super(Dropout, self).__init__()

        self.dist = dist
        assert self.dist in Dropout.DISTRIBUTIONS

        self.p = float(p)
        assert 0. <= self.p <= 1.

        self.mu = float(mu)
        self.sigma = float(sigma)
        assert self.sigma > 0.

    def forward(self, x):
        if self.dist == 'bernoulli':
            out = F.dropout(x, self.p, self.training)
        elif self.dist == 'gaussian':
            if self.training:
                with torch.no_grad():
                    soft_mask = x.new_empty(x.size()).normal_(self.mu, self.sigma).clamp_(0., 1.)

                scale = 1. / self.mu
                out = scale * soft_mask * x
            else:
                out = x
        else:
            out = x

        return out
