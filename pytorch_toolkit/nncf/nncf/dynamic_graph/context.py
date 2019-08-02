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

import logging
import threading
from collections import deque, namedtuple
from contextlib import contextmanager

import networkx as nx

from _warnings import warn

from .trace_tensor import TracedTensor, inputs_match, make_input_infos
from ..operator_names import get_version_agnostic_name

_CURRENT_CONTEXT = None
_ALL_CONTEXTS = {}

Hook = namedtuple('Hook', ['callback', 'allow_nested'])

logger = logging.getLogger(__name__)


class TracingContext:
    def __init__(self, name):
        self.name = name
        self.graph = nx.DiGraph()
        self.nodes = {}

        self._save_context = None
        self._post_hooks = {}
        self._pre_hooks = {}
        self._num_nested_hooks = 0

        self._thread_local = threading.local()

        self._n_instance = 0
        self._cond = threading.Condition()

    def _find_operator_node(self, inputs, operator_type, call_context):
        with self._cond:
            self._n_instance += 1

        nodes = self.graph.nodes

        node_candidates = self._find_consumer_nodes(nodes, call_context, inputs, operator_type)
        if not node_candidates:
            node_candidates = self._find_input_nodes(nodes, call_context, inputs, operator_type)

        result = None
        if len(node_candidates) == 1:
            result = next(iter(node_candidates.items()))[1]
        if len(node_candidates) > 1:
            warn("More than one node matches input")
            result = next(iter(node_candidates.items()))[1]

        with self._cond:
            self._n_instance -= 1
            self._cond.notify_all()

        return result

    def _add_node(self, inputs, operator_type, call_context):
        with self._cond:
            while self._n_instance > 0:
                self._cond.wait()

            #check that a node has already been added
            node = self._find_operator_node(inputs, operator_type, call_context)
            if node is not None:
                return node

            node_id = self._get_node_idx()
            node_name = self._get_node_name(node_id, operator_type)

            logger.debug("new_node: {} Scope: {}".format(node_name, str(self.scopes)))

            self.nodes[node_id] = node_name
            self.graph.add_node(
                node_name, id=node_id, type=operator_type,
                name=node_name, scope=self.scopes.copy(), context=call_context,
                input_infos=make_input_infos(inputs)
            )

            for input_ in inputs:
                if not isinstance(input_, TracedTensor):
                    continue
                parent = self.nodes[input_.tensor_meta.creator_id]
                self.graph.add_edge(parent, node_name)

        return self.graph.nodes[node_name]

    def find_operator_node(self, inputs, operator_type, call_context):
        version_agnostic_operator_type = get_version_agnostic_name(operator_type)
        if version_agnostic_operator_type is not None:
            operator_type = version_agnostic_operator_type

        node = self._find_operator_node(inputs, operator_type, call_context)
        if node is None:
            node = self._add_node(inputs, operator_type, call_context)
        return node

    def reset_scope_operator_call_counters(self):
        """
        Must be called after each "forward" operation of the model that is made
        within this context
        """
        self._thread_local.operator_counters = {}

    def register_scope_operator_call(self, operator_type, scope):
        scoped_op_name = '{}/{}'.format('/'.join(scope), operator_type)
        if scoped_op_name in self._thread_local.operator_counters:
            self._thread_local.operator_counters[scoped_op_name] += 1
        else:
            self._thread_local.operator_counters[scoped_op_name] = 1

    def get_operator_call_count_in_scope(self, operator_type, scope):
        scoped_op_name = '{}/{}'.format('/'.join(scope), operator_type)
        if scoped_op_name in self._thread_local.operator_counters:
            return self._thread_local.operator_counters[scoped_op_name]
        return 0

    def enter(self):
        global _CURRENT_CONTEXT
        self._save_context = _CURRENT_CONTEXT
        _CURRENT_CONTEXT = self

    def leave(self):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT = self._save_context
        self._save_context = None

    def push_scope(self, scope_module):
        scope_name = self._get_scope_name(scope_module)
        self.scope_modules.append(scope_module)
        self.scopes.append(scope_name)

    def pop_scope(self):
        self.scopes.pop()
        self.scope_modules.pop()

    def register_hook(self, fn, name=None, allow_nested=False, post=True):
        if name is None:
            name = fn.__name__
        if post and name in self._post_hooks:
            raise KeyError("Post hook with the name {} is already registered".format(name))
        if not post and name in self._pre_hooks:
            raise KeyError("Pre hook with the name {} is already registered".format(name))
        self._post_hooks[name] = Hook(fn, allow_nested)

    def execute_hooks(self, node, inputs, outputs):
        in_op = getattr(self, 'in_operator', False)
        self.in_operator = False
        self._thread_local.num_nested_hooks += 1
        for hook in self._post_hooks.values():
            if not hook.allow_nested and self._thread_local.num_nested_hooks > 1:
                continue
            hook_out = hook.callback(self, node, inputs, outputs)
            if hook_out is not None:
                outputs = hook_out
        self._thread_local.num_nested_hooks -= 1
        self.in_operator = in_op
        return outputs

    @property
    def replica(self):
        self._init_thread_local()
        return self._thread_local.replica

    @replica.setter
    def replica(self, value):
        self._init_thread_local()
        self._thread_local.replica = value

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
    def scopes(self):
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
        tl.replica = None
        tl.operator_counters = {}

    def _find_input_nodes(self, nodes, call_context, inputs, operator_type):
        node_candidates = {}
        for name, node in nodes.items():
            if self.graph.in_degree(name) != 0:
                continue
            if self._node_match(node, inputs, operator_type, call_context):
                node_candidates[name] = node
        return node_candidates

    def _node_match(self, node, inputs, operator_type, call_context):
        if node['type'] != operator_type:
            return False
        if call_context != node['context']:
            return False
        if self.scopes != node['scope']:
            return False
        if not inputs_match(node['input_infos'], inputs):
            return False
        return True

    def _find_consumer_nodes(self, nodes, call_context, inputs, operator_type):
        node_candidates = {}
        for i in inputs:
            if not isinstance(i, TracedTensor):
                continue
            creator_id = i.tensor_meta.creator_id
            for successor in self.graph.successors(self.nodes[creator_id]):
                successor_node = nodes[successor]
                if self._node_match(successor_node, inputs, operator_type, call_context):
                    node_candidates[successor] = successor_node
        return node_candidates

    def _get_scope_name(self, module):
        module_class = module.__class__.__name__
        if not self.scope_modules:
            return module_class
        q = deque([(tuple(), self.scope_modules[-1])])
        while q:
            name_parts, top = q.popleft()
            if module is top:
                return "/".join(name_parts)
            for name, child in top.named_children():
                child_name = "{cls}[{name}]".format(cls=child.__class__.__name__, name=name)
                q.append((name_parts + (child_name,), child))
        return module_class

    def _get_node_name(self, node_id, operator_name):
        if self.name:
            name_parts = (*self.scopes, operator_name)
        else:
            name_parts = (self.name, *self.scopes, operator_name)
        return '{id} {uri}'.format(uri='/'.join(name_parts), id=node_id)

    def _get_node_idx(self):
        return len(self.nodes)


def set_current_context(context_name=None):
    current_context = get_context(context_name)
    current_context.enter()
    return current_context


@contextmanager
def context(name):
    ctx = get_context(name)
    ctx.enter()
    yield ctx
    ctx.reset_scope_operator_call_counters()
    ctx.leave()


def get_context(context_name=None):
    if context_name in _ALL_CONTEXTS:
        return _ALL_CONTEXTS[context_name]
    _ALL_CONTEXTS[context_name] = TracingContext(context_name)
    return _ALL_CONTEXTS[context_name]


def reset_context(context_name=None):
    _ALL_CONTEXTS[context_name] = TracingContext(context_name)
    return _ALL_CONTEXTS[context_name]


def get_current_context():
    return _CURRENT_CONTEXT
