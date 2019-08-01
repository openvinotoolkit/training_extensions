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

import warnings

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from .context import get_current_context
from .trace_tensor import (TracedTensor, flatten_args, get_caller_context,
                           trace_tensors)

_IGNORED_SCOPES = []


def ignore_scope(cls):
    if cls not in _IGNORED_SCOPES:
        _IGNORED_SCOPES.append(cls)
    return cls


def register_operator(name=None):
    def wrap(operator):
        op_name = name
        if op_name is None:
            op_name = operator.__name__
        return wrap_operator(operator, op_name)

    return wrap


def wrap_operator(operator, operator_type):
    # do not wrap function twice
    _orig_op = getattr(operator, '_original_op', None)
    if _orig_op is not None:
        raise Exception("Operator: {} is already wrapped".format(_orig_op.__name__))

    def wrapped(*args, **kwargs):
        ctx = get_current_context()
        if not ctx or getattr(ctx, 'in_operator', False):
            op1 = operator(*args, **kwargs)
            return op1

        ctx.in_operator = True

        call_ctx = get_caller_context(operator_type, ctx)
        ctx.register_scope_operator_call(operator_type, ctx.scopes)
        fargs = flatten_args(args, kwargs)
        node = ctx.find_operator_node(fargs, operator_type, call_ctx)
        result = operator(*args, **kwargs)
        result = trace_tensors(result, node)
        result = ctx.execute_hooks(node, fargs, result)

        ctx.in_operator = False
        return result

    # pylint: disable=protected-access
    wrapped._original_op = operator
    return wrapped


def wrap_module_call(module_call):
    def wrapped(self, *args, **kwargs):
        ctx = get_current_context()
        if not ctx or self.__class__ in _IGNORED_SCOPES:
            if isinstance(self, DataParallel):
                _warn_data_parallel()
            return module_call(self, *args, **kwargs)
        ctx.push_scope(self)
        retval = module_call(self, *args, **kwargs)
        ctx.pop_scope()
        return retval

    return wrapped


def torch_jit_script_wrapper(*args, **kwargs):
    # Torch JIT cannot work with NNCF-modified operators,
    # so at each import of a @torch.jit.script-decorated
    # function we need to un-patch the torch operators
    unpatch_torch_operators()

    # This import statement is required, otherwise we get a
    # "RuntimeError: undefined value torch" inside the real torch.jit.script

    # pylint:disable=unused-import
    import torch

    retval = ORIGINAL_OPERATORS["script"](*args, **kwargs)
    patch_torch_operators()
    return retval


def _warn_data_parallel():
    if getattr(_warn_data_parallel, 'warned_once', False):
        return
    _warn_data_parallel.warned_once = True
    warnings.warn("You are using DataParallel, which may cause significant performance issues with dynamic graph "
                  "building. Consider using distributed training (DistributedDataParallel) instead")


PATCHED_OPERATORS = [
    "conv2d",
    "conv_transpose2d",
    "max_pool2d",
    "linear",
    "dropout",
    "threshold",
    "batch_norm",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "sigmoid",
    "softmax",
    "hardtanh",
]
CORE_OPERATORS = [
    "cat",
    "relu",
    "relu_",
    "max",
    "min",
]

TENSOR_OPERATORS = [
    "view",
    "permute",
    "contiguous",
    "reshape",
    "mean",
    "__iadd__",
    "__add__",
    "__imul__",
    "__mul__",
    "__idiv__",
    "__div__",
    "__truediv__",
    # "__getitem__",
    "round"
]


ORIGINAL_OPERATORS = {}
_JIT_ALREADY_WRAPPED = False
_OPERATORS_ALREADY_WRAPPED = False


def patch_torch_jit_script():
    import torch
    orig = getattr(torch.jit, "script")
    ORIGINAL_OPERATORS["script"] = orig
    setattr(torch.jit, "script", torch_jit_script_wrapper)


def patch_torch_operators():
    # Only patch torch.jit.script during first patch_torch_operators call
    global _JIT_ALREADY_WRAPPED
    if not _JIT_ALREADY_WRAPPED:
        patch_torch_jit_script()
        _JIT_ALREADY_WRAPPED = True

    # Do not patch operators twice as well
    global _OPERATORS_ALREADY_WRAPPED
    if _OPERATORS_ALREADY_WRAPPED:
        return
    _OPERATORS_ALREADY_WRAPPED = True

    # patch nn.functional operators
    import torch.nn.functional as F
    for op_name in PATCHED_OPERATORS:
        orig = getattr(F, op_name)
        ORIGINAL_OPERATORS[op_name] = orig
        setattr(F, op_name, wrap_operator(orig, op_name))

    # path core operators
    import torch
    for op_name in CORE_OPERATORS:
        orig = getattr(torch, op_name)
        ORIGINAL_OPERATORS[op_name] = orig
        setattr(torch, op_name, wrap_operator(orig, op_name))

    for op_name in TENSOR_OPERATORS:
        orig = getattr(TracedTensor, op_name)
        ORIGINAL_OPERATORS[op_name] = orig
        setattr(TracedTensor, op_name, wrap_operator(orig, op_name))

    ORIGINAL_OPERATORS["__call__"] = torch.nn.Module.__call__
    torch.nn.Module.__call__ = wrap_module_call(torch.nn.Module.__call__)
    ignore_scope(DataParallel)
    ignore_scope(DistributedDataParallel)


def unpatch_torch_operators():
    global _OPERATORS_ALREADY_WRAPPED
    if not _OPERATORS_ALREADY_WRAPPED:
        return
    _OPERATORS_ALREADY_WRAPPED = False

    # unpatch nn.functional operators
    import torch.nn.functional as F
    for op_name in PATCHED_OPERATORS:
        setattr(F, op_name, ORIGINAL_OPERATORS[op_name])

    # path core operators
    import torch
    for op_name in CORE_OPERATORS:
        setattr(torch, op_name, ORIGINAL_OPERATORS[op_name])

    for op_name in TENSOR_OPERATORS:
        setattr(TracedTensor, op_name, ORIGINAL_OPERATORS[op_name])

    torch.nn.Module.__call__ = ORIGINAL_OPERATORS["__call__"]
