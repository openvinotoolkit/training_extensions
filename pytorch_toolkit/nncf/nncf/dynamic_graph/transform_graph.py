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

from nncf.layers import NNCF_MODULES_DICT
from nncf.utils import get_node_name, in_scope_list


def replace_module_by_nncf_module(module: nn.Module):
    for nncf_module_type, module_type in NNCF_MODULES_DICT.items():
        if module.__class__.__name__ == module_type.__name__:
            nncf_module = module
            if not module.__class__.__name__ == nncf_module_type.__name__:
                nncf_module = nncf_module_type.from_module(module)
            return nncf_module
    return module


def replace_modules_by_nncf_modules(model: nn.Module, ignored_scopes=None, target_scopes=None, logger=None):
    replace_fn = partial(replace_module_by_nncf_module)
    return replace_modules(model, replace_fn, ignored_scopes=ignored_scopes, target_scopes=target_scopes, logger=logger)


def replace_modules(model: nn.Module, replace_fn, ignored_scopes=None, target_scopes=None, memo=None, prefix=None,
                    logger=None):
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
                if in_scope_list(child_name, ignored_scopes):
                    if logger is not None:
                        logger.info("Ignored wrapping modules in scope: {}".format(child_name))
                    continue

                if target_scopes is None or in_scope_list(child_name, target_scopes):
                    if logger is not None:
                        logger.info("Wrapping module {} by {}".format(
                            child_name, get_node_name(replaced_module, name, prefix)))
                    if isinstance(model, nn.Sequential):
                        # pylint: disable=protected-access
                        model._modules[name] = replaced_module
                    else:
                        setattr(model, name, replaced_module)

            replace_modules(module, replace_fn, ignored_scopes, target_scopes, memo, child_name, logger)
    return model
