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

import torch.nn as nn
import nncf.utils
from .layer_utils import _NNCFModuleMixin


class NNCFConv2d(_NNCFModuleMixin, nn.Conv2d):
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.Conv2d.__name__
        nncf_conv = NNCFConv2d(
            module.in_channels, module.out_channels, module.kernel_size, module.stride,
            module.padding, module.dilation, module.groups, hasattr(module, 'bias')
        )
        nncf.utils.dict_update(nncf_conv.__dict__, module.__dict__)
        return nncf_conv


class NNCFLinear(_NNCFModuleMixin, nn.Linear):
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.Linear.__name__

        nncf_linear = NNCFLinear(module.in_features, module.out_features, hasattr(module, 'bias'))
        nncf.utils.dict_update(nncf_linear.__dict__, module.__dict__)
        return nncf_linear


class NNCFConvTranspose2d(_NNCFModuleMixin, nn.ConvTranspose2d):
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.ConvTranspose2d.__name__
        args = [module.in_channels, module.out_channels, module.kernel_size, module.stride,
                module.padding, module.output_padding, module.groups, hasattr(module, 'bias'),
                module.dilation]
        if hasattr(module, 'padding_mode'):
            args.append(module.padding_mode)
        nncf_conv_transpose2d = NNCFConvTranspose2d(*args)
        nncf.utils.dict_update(nncf_conv_transpose2d.__dict__, module.__dict__)
        return nncf_conv_transpose2d


class NNCFConv3d(_NNCFModuleMixin, nn.Conv3d):
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.Conv3d.__name__

        nncf_conv3d = NNCFConv3d(
            module.in_channels, module.out_channels, module.kernel_size, module.stride,
            module.padding, module.dilation, module.groups, hasattr(module, 'bias')
        )
        nncf.utils.dict_update(nncf_conv3d.__dict__, module.__dict__)
        return nncf_conv3d


class NNCFConvTranspose3d(_NNCFModuleMixin, nn.ConvTranspose3d):
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.ConvTranspose3d.__name__
        args = [module.in_channels, module.out_channels, module.kernel_size, module.stride,
                module.padding, module.output_padding, module.groups, hasattr(module, 'bias'),
                module.dilation]
        if hasattr(module, 'padding_mode'):
            args.append(module.padding_mode)
        nncf_conv_transpose3d = NNCFConvTranspose3d(*args)
        nncf.utils.dict_update(nncf_conv_transpose3d.__dict__, module.__dict__)
        return nncf_conv_transpose3d


NNCF_MODULES_DICT = {
    NNCFConv2d: nn.Conv2d,
    NNCFLinear: nn.Linear,
    NNCFConvTranspose2d: nn.ConvTranspose2d,
    NNCFConv3d: nn.Conv3d,
    NNCFConvTranspose3d: nn.ConvTranspose3d,
}

NNCF_MODULES_MAP = {k.__name__: v.__name__ for k, v in NNCF_MODULES_DICT.items()}
NNCF_MODULES = list(NNCF_MODULES_MAP.keys())
