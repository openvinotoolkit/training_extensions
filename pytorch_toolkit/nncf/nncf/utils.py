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

import random
import warnings
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import torch
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
    checked_scope_stack = scope.split('/')
    for item in [scope_list] if isinstance(scope_list, str) else scope_list:
        scope_stack = item.split('/') if isinstance(item, str) else item
        if scopes_matched(checked_scope_stack, scope_stack):
            return True
    return False


def parse_node_name(name):
    slash_pos = -1
    nbrackets = 0
    for i, ch in enumerate(reversed(name)):
        if ch == ']':
            nbrackets += 1
        elif ch == '[':
            nbrackets -= 1
        elif ch == '/' and nbrackets == 0:
            slash_pos = len(name) - i - 1
            break

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
    ctx = build_graph(model, graph_scope, input_args=input_args, reset_context=True)
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


def get_all_modules_by_type(model, module_types, prefix=None, ignored_scopes=None, target_scopes=None):
    if isinstance(module_types, str):
        module_types = [module_types]
    found = OrderedDict()
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)

        if in_scope_list(full_node_name, ignored_scopes):
            continue

        if target_scopes is None or in_scope_list(full_node_name, target_scopes):
            if module_types.count(str(type(module).__name__)) != 0:
                found[full_node_name] = module
            sub_found = get_all_modules_by_type(module, module_types,
                                                prefix=full_node_name,
                                                ignored_scopes=ignored_scopes,
                                                target_scopes=target_scopes)
            if sub_found:
                found.update(sub_found)
    return found


def get_state_dict_names_with_modules(model, str_types=None, prefix=''):
    found = OrderedDict()
    for name, module in model.named_children():
        full_node_name = "{}{}".format(prefix, name)
        if str_types is not None and type(module).__name__ in str_types:
            found[full_node_name] = module
        sub_found = get_state_dict_names_with_modules(module, str_types, prefix=full_node_name + '.')
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


def sum_like(tensor_to_sum, ref_tensor):
    """Warning: may modify tensor_to_sum"""
    if ref_tensor.size == 1:
        return tensor_to_sum.sum()

    for dim, size in enumerate(ref_tensor.shape):
        if size == 1:
            if isinstance(tensor_to_sum, np.ndarray):
                tensor_to_sum = tensor_to_sum.sum(dim, keepdims=True)
            else:
                tensor_to_sum = tensor_to_sum.sum(dim, keepdim=True)
    return tensor_to_sum


def get_per_channel_scale_shape(input_shape, is_weights):
    scale_shape = [1 for _ in input_shape]
    if is_weights:
        scale_shape[0] = input_shape[0]  # Per weight channel scales
    else:
        scale_shape[1] = input_shape[1]  # Per activation channel scales

    elements = 1
    for i in scale_shape:
        elements *= i
    if elements == 1:
        return 1

    return scale_shape
