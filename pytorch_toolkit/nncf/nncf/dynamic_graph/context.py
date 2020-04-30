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

import re
import threading
from collections import deque
from contextlib import contextmanager
from typing import Callable, List
from copy import deepcopy

from nncf.debug import is_debug
from nncf.dynamic_graph.graph import InputAgnosticOperationExecutionContext
from nncf.dynamic_graph.graph import NNCFGraph, NNCFNode
from nncf.dynamic_graph.trace_tensor import make_input_infos
from nncf.dynamic_graph.version_agnostic_op_names import get_version_agnostic_name

_CURRENT_CONTEXT = None


class OperatorInput:
    def __init__(self, op_args, op_kwargs):
        self.op_args = op_args
        self.op_kwargs = op_kwargs


class ScopeElement:
    def __init__(self, calling_module_class_name: str, calling_field_name: str = None):
        self.calling_module_class_name = calling_module_class_name
        self.calling_field_name = calling_field_name

    def __str__(self):
        if self.calling_field_name is None:
            return self.calling_module_class_name
        return "{cls}[{name}]".format(cls=self.calling_module_class_name,
                                      name=self.calling_field_name)

    def __eq__(self, other: 'ScopeElement'):
        return (self.calling_module_class_name == other.calling_module_class_name) and \
               (self.calling_field_name == other.calling_field_name)

    def __hash__(self):
        return hash((self.calling_module_class_name, self.calling_field_name))

    @staticmethod
    def from_str(string: str):
        matches = re.search(r"(.*)\[(.*)\]|(.*)", string)
        if matches is None:
            raise RuntimeError("Invalid scope element string")
        if matches.groups()[0] is None and matches.groups()[1] is None:
            return ScopeElement(matches.groups()[2])
        if matches.groups()[0] is not None and matches.groups()[1] is not None:
            return ScopeElement(matches.groups()[0], matches.groups()[1])
        raise RuntimeError("Could not parse the scope element string")


class Scope:
    def __init__(self, scope_elements: List[ScopeElement] = None):
        if scope_elements is not None:
            self.scope_elements = scope_elements
        else:
            self.scope_elements = []

    def __str__(self):
        return '/'.join([str(scope_el) for scope_el in self.scope_elements])

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: 'Scope'):
        return self.scope_elements == other.scope_elements

    def __getitem__(self, key):
        return self.scope_elements[key]

    def __contains__(self, item: 'Scope'):
        """Idiom: ('A/B/C' in 'A/B') == True"""
        if len(self.scope_elements) > len(item.scope_elements):
            return False
        for i in range(len(self.scope_elements)):
            if self.scope_elements[i] != item.scope_elements[i]:
                return False
        return True

    def __add__(self, rhs):
        init_list = self.scope_elements + rhs.scope_elements
        return Scope(init_list)

    def copy(self):
        return Scope(deepcopy(self.scope_elements))

    def push(self, scope_element: ScopeElement):
        self.scope_elements.append(scope_element)

    def pop(self) -> ScopeElement:
        return self.scope_elements.pop()

    @staticmethod
    def from_str(string: str) -> 'Scope':
        elts = string.split('/')
        return Scope([ScopeElement.from_str(s) for s in elts])


# pylint: disable=too-many-public-methods
class TracingContext:
    def __init__(self):
        self.graph = NNCFGraph()

        self._save_context = None
        self._post_hooks = {}
        self._pre_hooks = {}
        self._num_nested_hooks = 0

        self._thread_local = threading.local()

        self._n_instance = 0
        self._cond = threading.Condition()
        self.is_tracing = True
        self._input_comparators_per_scope = []

    def __enter__(self):
        global _CURRENT_CONTEXT
        self._save_context = _CURRENT_CONTEXT
        _CURRENT_CONTEXT = self
        self._init_thread_local()
        if is_debug():
            self.reset_node_call_counters()

        return self

    def __exit__(self, *args):
        self.reset_scope_operator_call_counters()
        self.leave()

    def find_operator_node(self, inputs,
                           ia_op_exec_context: InputAgnosticOperationExecutionContext) -> NNCFNode:
        with self._cond:
            self._n_instance += 1
        tensor_metas = make_input_infos(inputs)

        node = self.graph.find_node(ia_op_exec_context, tensor_metas, self._input_comparators_per_scope)

        with self._cond:
            self._n_instance -= 1
            self._cond.notify_all()

        if node is None:
            with self._cond:
                while self._n_instance > 0:
                    self._cond.wait()
                # Another thread may have added a node inside this block,
                # so we need to check again if a node is already added.
                node = self.graph.find_node(ia_op_exec_context, tensor_metas, self._input_comparators_per_scope)
                if node is None:
                    node = self.graph.add_node(ia_op_exec_context, tensor_metas, self._input_comparators_per_scope,
                                               inputs)
        return node

    def get_caller_context(self, operator_type: str) -> InputAgnosticOperationExecutionContext:
        """
        Designed to work in the following way - for each scope the context will track the number of the calls to the
        operators with the name operator_type (call_order). The counter values are preserved until reset by a
        corresponding member function of the context, which must be called after each model iteration - this is
        usually handled inside NNCF. This mechanism allows to discern between multiple function calls inside the same
        module that would each require their own instance of compression layers - for instance, multiple `relu`
        function calls (either on their own or inside a `for` cycle), and at the same moment allow the checkpoints to
        be loaded if the model had changed in the meantime in a way that does not impact the major function call
        order (e.g. if comments were added to the .py file with the model)
        """
        version_agnostic_operator_type = get_version_agnostic_name(operator_type)

        call_order = self.get_operator_call_count_in_scope(version_agnostic_operator_type, self.scope)

        ia_op_exec_context = InputAgnosticOperationExecutionContext(version_agnostic_operator_type,
                                                                    self.scope,
                                                                    call_order)
        return ia_op_exec_context

    def reset_scope_operator_call_counters(self):
        """
        Must be called after each "forward" operation of the model that is made
        within this context
        """
        self._thread_local.operator_counters = {}

    @staticmethod
    def _get_operator_counter_key(operator_name: str, scope: Scope):
        return "{}_{}".format(str(scope), operator_name)

    def register_operator_call(self, operator_name: str, scope: Scope):
        key = self._get_operator_counter_key(operator_name, scope)
        if key in self._thread_local.operator_counters:
            self._thread_local.operator_counters[key] += 1
        else:
            self._thread_local.operator_counters[key] = 1

    def get_operator_call_count_in_scope(self, operator_name: str, scope: Scope):
        key = self._get_operator_counter_key(operator_name, scope)
        if key in self._thread_local.operator_counters:
            return self._thread_local.operator_counters[key]
        return 0

    def reset_operator_call_count_in_scope(self, scope):
        scoped_op_name = str(scope)
        for key in self._thread_local.operator_counters.keys():
            if scoped_op_name in key:
                self._thread_local.operator_counters[key] = 0

    def enter(self):
        global _CURRENT_CONTEXT
        self._save_context = _CURRENT_CONTEXT
        _CURRENT_CONTEXT = self
        self._init_thread_local()

    def leave(self):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT = self._save_context
        self._save_context = None

    def push_scope(self, scope_module):
        relative_scopes_list = self._get_scope_relative_to_last_registered_module_call(scope_module)
        self.scope_modules.append(scope_module)
        self.relative_scopes_stack.append(relative_scopes_list)

    def pop_scope(self):
        self.relative_scopes_stack.pop()
        self.scope_modules.pop()

    def register_pre_hooks(self, fn_list: List[Callable], ia_op_exec_context: InputAgnosticOperationExecutionContext):
        if ia_op_exec_context in self._pre_hooks:
            raise KeyError("Pre hook for context {} is already registered".format(str(ia_op_exec_context)))
        self._pre_hooks[ia_op_exec_context] = fn_list

    def execute_pre_hooks(self, ia_op_exec_context: InputAgnosticOperationExecutionContext,
                          op_inputs: OperatorInput) -> OperatorInput:
        in_op = getattr(self, 'in_operator', False)
        self.in_operator = False
        self._thread_local.num_nested_hooks += 1
        if ia_op_exec_context in self._pre_hooks:
            for hook in self._pre_hooks[ia_op_exec_context]:
                op_inputs = hook(op_inputs)
        self._thread_local.num_nested_hooks -= 1
        self.in_operator = in_op
        return op_inputs

    def register_post_hooks(self, fn_list: List[Callable], ia_op_exec_context: InputAgnosticOperationExecutionContext):
        if ia_op_exec_context in self._post_hooks:
            raise KeyError("Post hook for context {} is already registered".format(str(ia_op_exec_context)))
        self._post_hooks[ia_op_exec_context] = fn_list

    def execute_post_hooks(self, ia_op_exec_context: InputAgnosticOperationExecutionContext, outputs):
        in_op = getattr(self, 'in_operator', False)
        self.in_operator = False
        self._thread_local.num_nested_hooks += 1
        if ia_op_exec_context in self._post_hooks:
            for hook in self._post_hooks[ia_op_exec_context]:
                outputs = hook(outputs)
        self._thread_local.num_nested_hooks -= 1
        self.in_operator = in_op
        return outputs

    def disable_tracing(self):
        self.is_tracing = False

    def enable_tracing(self):
        self.is_tracing = True

    def add_node_comparators(self, scopes_to_apply: List[str],
                             node_input_comparator: 'TensorMetaComparator' = None):
        self._input_comparators_per_scope.append((node_input_comparator, scopes_to_apply))

    @property
    def base_module_thread_local_replica(self):
        self._init_thread_local()
        return self._thread_local.base_module_replica

    @base_module_thread_local_replica.setter
    def base_module_thread_local_replica(self, value):
        self._init_thread_local()
        self._thread_local.base_module_replica = value

    @property
    def in_operator(self):
        self._init_thread_local()
        return self._thread_local.in_operator

    @in_operator.setter
    def in_operator(self, val):
        self._init_thread_local()
        self._thread_local.in_operator = val

    @property
    def scope_modules(self):
        self._init_thread_local()
        return self._thread_local.scope_modules

    @property
    def relative_scopes_stack(self) -> List[Scope]:
        self._init_thread_local()
        return self._thread_local.scopes

    def _init_thread_local(self):
        # todo: master node part!
        tl = self._thread_local
        if getattr(tl, 'ready', False):
            return
        tl.ready = True
        tl.scopes = []
        tl.scope_modules = []
        tl.in_operator = False
        tl.num_nested_hooks = 0
        tl.base_module_replica = None
        tl.operator_counters = {}
        tl.node_call_tracker = {}

    def register_node_call(self, node_key: str):
        if node_key in self._thread_local.node_call_tracker:
            self._thread_local.node_call_tracker[node_key] += 1
        else:
            self._thread_local.node_call_tracker[node_key] = 1

    def reset_node_call_counters(self):
        for k, _ in self._thread_local.node_call_tracker.items():
            self._thread_local.node_call_tracker[k] = 0

    def get_node_call_counter_dict(self):
        return self._thread_local.node_call_tracker

    def _get_scope_relative_to_last_registered_module_call(self, module) -> Scope:
        module_class = module.__class__.__name__
        if not self.scope_modules:
            return Scope([ScopeElement(module_class), ])
        q = deque([(tuple(), self.scope_modules[-1])])
        while q:
            scope_parts, top = q.popleft()
            if module is top:
                return Scope(list(scope_parts))
            for name, child in top.named_children():
                scope_element = ScopeElement(child.__class__.__name__, name)
                q.append((scope_parts + (scope_element,), child))
        return Scope([ScopeElement(module_class), ])

    @property
    def scope(self) -> Scope:
        stack_copy = self.relative_scopes_stack.copy()
        scope_el_list = []
        for relative_scope in stack_copy:
            for scope_element in relative_scope.scope_elements:
                scope_el_list.append(scope_element)
        return Scope(scope_el_list)

    def reset_graph(self):
        self.graph = NNCFGraph()


@contextmanager
def no_nncf_trace():
    ctx = get_current_context()
    if ctx is not None:
        ctx.disable_tracing()
    yield
    if ctx is not None:
        ctx.enable_tracing()


def get_current_context() -> TracingContext:
    return _CURRENT_CONTEXT
