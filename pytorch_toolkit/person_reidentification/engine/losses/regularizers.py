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
import torch.nn as nn


class ConvRegularizer(nn.Module):
    def __init__(self, reg_class, controller):
        super().__init__()
        self.reg_instance = reg_class(controller)

    def get_all_conv_layers(self, module):
        if isinstance(module, (nn.Sequential, list)):
            for m in module:
                yield from self.get_all_conv_layers(m)

        if isinstance(module, nn.Conv2d):
            yield module

    def forward(self, net, ignore=False):

        accumulator = torch.tensor(0.0).cuda()

        if ignore:
            return accumulator

        all_mods = [module for module in net.module.modules() if type(module) != nn.Sequential]
        for conv in self.get_all_conv_layers(all_mods):
            accumulator += self.reg_instance(conv.weight)

        return accumulator


class SVMORegularizer(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def dominant_eigenvalue(self, A):  # A: 'N x N'
        N, _ = A.size()
        x = torch.rand(N, 1, device='cuda')
        Ax = (A @ x)
        AAx = (A @ Ax)
        return AAx.permute(1, 0) @ Ax / (Ax.permute(1, 0) @ Ax)

    def get_singular_values(self, A):  # A: 'M x N, M >= N'
        ATA = A.permute(1, 0) @ A
        N, _ = ATA.size()
        largest = self.dominant_eigenvalue(ATA)
        I = torch.eye(N, device='cuda')  # noqa
        I = I * largest  # noqa
        tmp = self.dominant_eigenvalue(ATA - I)
        return tmp + largest, largest

    def forward(self, W):  # W: 'S x C x H x W'
        # old_W = W
        old_size = W.size()
        if old_size[0] == 1:
            return 0
        W = W.view(old_size[0], -1).permute(1, 0)  # (C x H x W) x S
        smallest, largest = self.get_singular_values(W)
        return (
            self.beta * 10 * (largest - smallest)**2
        ).squeeze()


class NoneRegularizer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, _):
        return torch.tensor(0.0).cuda()


mapping = {
    False: NoneRegularizer,
    True: SVMORegularizer,
}


def get_regularizer(cfg_reg):
    name = cfg_reg.ow
    return ConvRegularizer(mapping[name], cfg_reg.ow_beta)
