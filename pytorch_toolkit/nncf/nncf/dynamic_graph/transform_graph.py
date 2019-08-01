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

from functools import partial

from torch import nn
from nncf.layers import NNCFConv2d, NNCFLinear, NNCFConvTranspose2d
from nncf.utils import dict_update, get_node_name, in_scope_list


def create_nncf_conv2d(module):
    assert module.__class__.__name__ == nn.Conv2d.__name__

    nncf_conv = NNCFConv2d(
        module.in_channels, module.out_channels, module.kernel_size, module.stride,
        module.padding, module.dilation, module.groups, hasattr(module, 'bias')
    )
    dict_update(nncf_conv.__dict__, module.__dict__)
    return nncf_conv


def create_nncf_linear(module):
    assert module.__class__.__name__ == nn.Linear.__name__

    nncf_linear = NNCFLinear(module.in_features, module.out_features, hasattr(module, 'bias'))
    dict_update(nncf_linear.__dict__, module.__dict__)
    return nncf_linear


def create_nncf_conv_transpose2d(module):
    assert module.__class__.__name__ == nn.ConvTranspose2d.__name__
    args = [module.in_channels, module.out_channels, module.kernel_size, module.stride,
            module.padding, module.output_padding, module.groups, hasattr(module, 'bias'),
            module.dilation]
    if hasattr(module, 'padding_mode'):
        args.append(module.padding_mode)
    nncf_conv_transpose2d = NNCFConvTranspose2d(*args)
    dict_update(nncf_conv_transpose2d.__dict__, module.__dict__)
    return nncf_conv_transpose2d


def replace_module_by_nncf_module(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        conv = module
        if not isinstance(module, NNCFConv2d):
            conv = create_nncf_conv2d(module)
        return conv
    if isinstance(module, nn.Linear):
        linear = module
        if not isinstance(module, NNCFLinear):
            linear = create_nncf_linear(module)
        return linear
    if isinstance(module, nn.ConvTranspose2d):
        conv_transpose2d = module
        if not isinstance(module, NNCFConvTranspose2d):
            conv_transpose2d = create_nncf_conv_transpose2d(module)
        return conv_transpose2d
    return module


def replace_modules_by_nncf_modules(model: nn.Module, ignored_scope=None, logger=None):
    replace_fn = partial(replace_module_by_nncf_module)
    return replace_modules(model, replace_fn, ignored_scope=ignored_scope, logger=logger)


def replace_modules(model: nn.Module, replace_fn, ignored_scope=None, memo=None, prefix=None, logger=None):
    if memo is None:
        memo = set()
        prefix = model.__class__.__name__

    if model not in memo:
        memo.add(model)
        for name, module in model.named_children():
            if module is None:
                continue

            child_name = get_node_name(module, name, prefix)
            replaced_module = replace_fn(module)

            if replaced_module is not None and module is not replaced_module:
                if ignored_scope and in_scope_list(child_name, ignored_scope):
                    if logger is not None:
                        logger.info("Ignored wrapping modules in scope: {}".format(child_name))
                    continue

                if logger is not None:
                    logger.info("Wrapping module {} by {}".format(
                        child_name, get_node_name(replaced_module, name, prefix)))
                if isinstance(model, nn.Sequential):
                    # pylint: disable=protected-access
                    model._modules[name] = replaced_module
                else:
                    setattr(model, name, replaced_module)

            replace_modules(module, replace_fn, ignored_scope, memo, child_name, logger)
    return model
