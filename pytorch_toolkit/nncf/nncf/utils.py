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

from collections import OrderedDict
from contextlib import contextmanager

import random
import numpy as np
import torch
import warnings
from texttable import Texttable

from .dynamic_graph.utils import build_graph
from .layers import NNCF_MODULES_MAP


def scopes_matched(scope_stack_0, scope_stack_1):
    if len(scope_stack_1) > len(scope_stack_0):
        return False

    for name0, name1 in zip(scope_stack_0, scope_stack_1):
        if name0 != name1:
            _, m_cls0, m_name0 = parse_node_name(name0)
            _, m_cls1, m_name1 = parse_node_name(name1)
            if m_name0 != m_name1 or not m_cls0 in NNCF_MODULES_MAP or m_cls1 != NNCF_MODULES_MAP[m_cls0]:
                scope = scope_stack_0[1:]
                if scope:
                    _, m_cls, _ = parse_node_name(scope[0])
                    scope[0] = m_cls
                    return scopes_matched(scope, scope_stack_1)
                return False
    return True


def in_scope_list(scope, scope_list):
    if scope_list is None:
        return False
    checked_scope_stack = scope.split('/') if isinstance(scope, str) else scope
    for item in [scope_list] if isinstance(scope_list, str) else scope_list:
        scope_stack = item.split('/') if isinstance(item, str) else item
        if scopes_matched(checked_scope_stack, scope_stack):
            return True
    return False


def parse_node_name(name):
    slash_pos = name.rfind('/')
    prefix = None if slash_pos < 0 else name[:slash_pos]

    last_name = name[slash_pos + 1:]
    open_bracket_pos = last_name.find("[")
    if open_bracket_pos < 0:
        return prefix, last_name, None
    return prefix, last_name[:open_bracket_pos], last_name[open_bracket_pos + 1:-1]


def get_node_name(module, module_name, prefix):
    return "{prefix}/{cls}[{name}]".format(prefix=prefix, cls=module.__class__.__name__, name=module_name)


def get_node_names_from_context(ctx):
    return [node_name.split(' ', 1)[1] for node_name in ctx.nodes.values()]


def get_all_node_names(model, input_sample_size, graph_scope=None):
    if graph_scope is None:
        graph_scope = 'utils'
    input_args = (next(model.parameters()).new_empty(input_sample_size),)
    ctx = build_graph(model, input_args, {}, graph_scope, reset_context=True)
    return get_node_names_from_context(ctx)


def get_all_modules(model, prefix=None):
    found = OrderedDict()
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)
        found[full_node_name] = module
        sub_found = get_all_modules(module, prefix=full_node_name)
        if sub_found:
            found.update(sub_found)
    return found


def get_all_modules_by_type(model, module_types, prefix=None, ignored_scope=None):
    if isinstance(module_types, str):
        module_types = [module_types]
    found = OrderedDict()
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)

        if ignored_scope and in_scope_list(full_node_name, ignored_scope):
            continue

        if module_types.count(str(type(module).__name__)) != 0:
            found[full_node_name] = module
        sub_found = get_all_modules_by_type(module, module_types, prefix=full_node_name, ignored_scope=ignored_scope)
        if sub_found:
            found.update(sub_found)
    return found


def set_module_by_node_name(model, node_name, module_to_set, prefix=None):
    if prefix is None:
        prefix = model.__class__.__name__

    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)
        if full_node_name == node_name:
            # pylint: disable=protected-access
            model._modules[name] = module_to_set
        set_module_by_node_name(module, node_name, module_to_set, full_node_name)


def get_module_by_node_name(model, node_name, prefix=None):
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)
        if full_node_name == node_name:
            return module
        sub_result = get_module_by_node_name(module, node_name, full_node_name)
        if sub_result is not None:
            return sub_result
    return None


def apply_by_node_name(model, node_names, command=lambda x: x, prefix=None):
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        node_name = get_node_name(module, name, prefix)
        if node_name in node_names:
            command(module)
        apply_by_node_name(module, node_names=node_names, command=command, prefix=node_name)


def print_statistics(stats):
    for key, val in stats.items():
        if isinstance(val, Texttable):
            print(key)
            print(val.draw())
        else:
            print("{}: {}".format(key, val))


def manual_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def is_tracing_state():
    # pylint: disable=protected-access
    return torch._C._get_tracing_state()


@contextmanager
def no_jit_trace():
    # pylint: disable=protected-access
    disable_tracing = torch.jit._disable_tracing()
    disable_tracing.__enter__()
    yield disable_tracing
    disable_tracing.__exit__()


def dict_update(src, dst, recursive=True):
    for name, value in dst.items():
        if recursive and name in src and isinstance(value, dict):
            dict_update(src[name], value, recursive)
        else:
            src[name] = value


def check_for_quantization_before_sparsity(child_algos: list):
    from nncf.sparsity.base_algo import BaseSparsityAlgo
    from nncf.quantization.algo import Quantization
    for idx, algo in enumerate(child_algos):
        if idx == len(child_algos) - 1:
            return
        if isinstance(algo, Quantization) and isinstance(child_algos[idx + 1], BaseSparsityAlgo):
            warnings.warn("Applying quantization before sparsity may lead to incorrect "
                          "sparsity metrics calculation. Consider revising the config file to specify "
                          "sparsity algorithms before quantization ones", RuntimeWarning)
