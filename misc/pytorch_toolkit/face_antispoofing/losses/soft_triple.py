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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


class SoftTripleLinear(nn.Module):
    """Computes similarities between input vectors and weights vectors"""
    def __init__(self, in_features, out_features, num_proxies=2, gamma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features*num_proxies))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.num_proxies = num_proxies
        self.gamma = 1. / gamma

    def forward(self, x):
        proxies_norm = F.normalize(self.weight, dim=0)
        similarities = F.normalize(x, dim=1).mm(proxies_norm)
        similarities = similarities.reshape(-1, self.out_features, self.num_proxies)
        prob = F.softmax(similarities * self.gamma, dim=2)
        sim_class = torch.sum(prob * similarities, dim=2)

        if self.traininig:
            return sim_class, proxies_norm.t().matmul(proxies_norm)

        return sim_class


class SoftTripleLoss(nn.Module):
    def __init__(self, cN, K, s=30, tau=.2, m=0.35):
        super().__init__()
        self.s = s
        self.tau = tau
        self.m = m
        self.cN = cN
        self.K = K
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1

    def forward(self, input_, target):
        ''' target - one hot '''
        # fold one_hot to one vector [batch size] (need to do it when label smooth or augmentations used)
        fold_target = target.argmax(dim=1)
        simClass, simCenter = input_
        pred = F.log_softmax(self.s * (simClass - F.one_hot(fold_target, simClass.shape[1]) * self.m), dim=-1)
        lossClassify = torch.mean(torch.sum(-target * pred, dim=-1))

        if self.tau > 0 and self.K > 1:
            reg = torch.sum(torch.sqrt(2.0 + 1e-5 - 2.*simCenter[self.weight])) / (self.cN * self.K * (self.K - 1.))
            return lossClassify + self.tau * reg
        else:
            return lossClassify
