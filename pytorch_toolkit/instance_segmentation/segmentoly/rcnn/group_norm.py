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


class GroupNormFunctionStub(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, num_groups, weight, bias, eps):
        return g.op('ExperimentalDetectronGroupNorm', input, weight, bias, num_groups_i=num_groups, eps_f=eps)

    @staticmethod
    def forward(ctx, input, num_groups, weight, bias, eps):
        return input


group_norm_stub = GroupNormFunctionStub.apply


class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps, affine)
        self.use_stub = False

    def forward(self, input):
        if not self.use_stub:
            return super().forward(input)
        else:
            return group_norm_stub(input, self.num_groups, self.weight, self.bias, self.eps)
