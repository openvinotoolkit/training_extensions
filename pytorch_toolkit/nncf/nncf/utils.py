"""
 Copyright (c) 2019-2020 Intel Corporation
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
from collections import OrderedDict
from contextlib import contextmanager
import re
from typing import Dict, Callable, Any, Mapping, Sequence, Set

import numpy as np
import torch
from torch import distributed as dist
from torch.nn import Module

from nncf.dynamic_graph.graph_builder import GraphBuilder, ModelInputInfo, create_dummy_forward_fn


def scopes_matched(scope_stack_0, scope_stack_1):
    from nncf.layers import NNCF_MODULES_MAP
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
        if "{re}" in item:
            regex = item.replace("{re}", "")
            if re.search(regex, scope):
                return True
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


def get_all_node_names(model, input_sample_size, builder=None):
    if not builder:
        builder = GraphBuilder(create_dummy_forward_fn([ModelInputInfo(input_sample_size), ]))
    graph = builder.build_graph(model)
    return [node_name.split(' ', 1)[1] for node_name in graph.get_all_node_keys()]


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


def get_all_modules_by_type(model, module_types, current_scope=None,
                            ignored_scopes=None, target_scopes=None) -> Dict['Scope', Module]:
    if isinstance(module_types, str):
        module_types = [module_types]
    found = OrderedDict()
    from nncf.dynamic_graph.context import Scope
    from nncf.dynamic_graph.context import ScopeElement
    if current_scope is None:
        current_scope = Scope()
        current_scope.push(ScopeElement(model.__class__.__name__))
    for name, module in model.named_children():
        child_scope_element = ScopeElement(module.__class__.__name__, name)
        child_scope = current_scope.copy()
        child_scope.push(child_scope_element)

        if in_scope_list(str(child_scope), ignored_scopes):
            continue

        if target_scopes is None or in_scope_list(str(child_scope), target_scopes):
            if module_types.count(str(type(module).__name__)) != 0:
                found[child_scope] = module
            sub_found = get_all_modules_by_type(module, module_types,
                                                current_scope=child_scope,
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


def get_module_by_node_name(model: torch.nn.Module, node_scope_str: str, prefix=None) -> torch.nn.Module:
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)
        if full_node_name == node_scope_str:
            return module
        sub_result = get_module_by_node_name(module, node_scope_str, full_node_name)
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


def get_flat_tensor_contents_string(input_tensor):
    retval = "["
    for idx, el in enumerate(input_tensor.view(-1)):
        if idx >= 10:
            retval += "... (first 10/{} elements shown only) ".format(len(input_tensor.view(-1)))
            break
        retval += "{:.4f}, ".format(el.item())
    retval += "]"
    return retval


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def safe_thread_call(main_call_fn, after_barrier_call_fn=None):
    result = None
    if is_dist_avail_and_initialized():
        if is_main_process():
            result = main_call_fn()
        dist.barrier()
        if not is_main_process():
            result = after_barrier_call_fn() if after_barrier_call_fn else main_call_fn()
    else:
        result = main_call_fn()
    return result

string_types = (str, bytes)
iteritems = lambda mapping: getattr(mapping, 'iteritems', mapping.items)()


def objwalk(obj, unary_predicate: Callable[[Any], bool], apply_fn: Callable, memo=None):
    if memo is None:
        memo = set()

    is_tuple = isinstance(obj, tuple)
    if is_tuple:
        obj = list(obj)

    def maybe_get_iterator(obj):
        it = None
        if isinstance(obj, Mapping):
            it = iteritems
        elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, string_types):
            it = enumerate
        return it

    iterator = maybe_get_iterator(obj)

    if iterator is not None:
        if id(obj) not in memo:
            memo.add(id(obj))
            indices_to_apply_fn_to = set()
            indices_vs_tuples_to_assign = {}  # type: Dict[Any, list]
            for idx, value in iterator(obj):
                next_level_it = maybe_get_iterator(value)
                if next_level_it is None:
                    if unary_predicate(value):
                        indices_to_apply_fn_to.add(idx)
                else:
                    if isinstance(value, tuple):
                        processed_tuple = objwalk(value, unary_predicate, apply_fn, memo)
                        indices_vs_tuples_to_assign[idx] = processed_tuple
                    else:
                        objwalk(value, unary_predicate, apply_fn)
            for idx in indices_to_apply_fn_to:
                obj[idx] = apply_fn(obj[idx])
            for idx, tpl in indices_vs_tuples_to_assign.items():
                obj[idx] = tuple(tpl)

            memo.remove(id(obj))
    else:
        if unary_predicate(obj):
            return apply_fn(obj)

    if is_tuple:
        return tuple(obj)

    return obj
