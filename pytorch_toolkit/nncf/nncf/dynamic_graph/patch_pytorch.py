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
from typing import Callable

from torch import Tensor
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from nncf.debug import is_debug
from nncf.dynamic_graph.context import get_current_context, OperatorInput
from nncf.dynamic_graph.graph import ITERATION_MODULES
from nncf.dynamic_graph.trace_tensor import TracedTensor, flatten_args, trace_tensors

_IGNORED_SCOPES = []


class CustomTraceFunction:
    def __call__(self, operator: Callable, *args, **kwargs):
        raise NotImplementedError


class ForwardTraceOnly(CustomTraceFunction):
    def __call__(self, operator: Callable, *args, **kwargs):
        """ This wrapper override will result in the operator not being added to graph,
        but the result will still have TracedTensors with parent IDs left the same as in input.
        Useful for operators which are not likely to be present in patterns considered for
        compression, but still have to be accounted for so that the NNCF internal graph representation
        does not become disjoint. """

        result = operator(*args, **kwargs)

        fargs = flatten_args(args, kwargs)
        input_traced_tensor_indices = [i for i in range(len(fargs)) if isinstance(fargs[i], TracedTensor)]

        if isinstance(result, (list, tuple)):
            output_tensors_to_be_traced_indices = [i for i in range(len(result)) if
                                                   isinstance(result[i], Tensor)]

            was_tuple = isinstance(result, tuple)
            result = list(result)
            if len(input_traced_tensor_indices) == 1:
                # Broadcast one and the same creator ID of input to all outputs
                for out_idx in output_tensors_to_be_traced_indices:
                    result[out_idx] = TracedTensor.from_torch_tensor(result[out_idx],
                                                                     fargs[input_traced_tensor_indices[
                                                                         0]].tensor_meta)
            elif len(input_traced_tensor_indices) != len(output_tensors_to_be_traced_indices):
                raise RuntimeError("Unable to forward trace through operator {} - "
                                   "input and output tensor count mismatch!".format(operator.__name__))
            else:
                # Assume that output tensor order corresponds to input tensor order
                for in_idx, out_idx in zip(input_traced_tensor_indices, output_tensors_to_be_traced_indices):
                    result[out_idx] = TracedTensor.from_torch_tensor(result[out_idx],
                                                                     fargs[in_idx].tensor_meta)
            if was_tuple:
                result = tuple(result)
        elif len(input_traced_tensor_indices) > 1:
            raise RuntimeError("Unable to forward trace through operator {} - "
                               "input and output tensor count mismatch!".format(operator.__name__))
        elif input_traced_tensor_indices:
            return TracedTensor.from_torch_tensor(result,
                                                  fargs[input_traced_tensor_indices[0]].tensor_meta)
        # No traced tensors in input, return a usual torch.Tensor as well
        return result


class PatchedOperatorInfo:
    def __init__(self, name: str, custom_trace_fn: CustomTraceFunction = None):
        """custom_trace_fn will be called instead of the regular node search/insertion step
        during the corresponding operator call"""
        self.name = name
        self.custom_trace_fn = custom_trace_fn


def ignore_scope(cls):
    if cls not in _IGNORED_SCOPES:
        _IGNORED_SCOPES.append(cls)
    return cls


def register_operator(name=None):
    def wrap(operator):
        op_name = name
        if op_name is None:
            op_name = operator.__name__
        return wrap_operator(operator, PatchedOperatorInfo(op_name))

    return wrap


def wrap_operator(operator, operator_info: PatchedOperatorInfo):
    # do not wrap function twice
    _orig_op = getattr(operator, '_original_op', None)
    if _orig_op is not None:
        raise Exception("Operator: {} is already wrapped".format(_orig_op.__name__))

    def wrapped(*args, **kwargs):
        ctx = get_current_context()
        if not ctx or getattr(ctx, 'in_operator', False) or not ctx.is_tracing:
            op1 = operator(*args, **kwargs)
            return op1

        ctx.in_operator = True

        if operator_info.custom_trace_fn is not None:
            result = operator_info.custom_trace_fn(operator, *args, **kwargs)
        else:
            ia_op_exec_context = ctx.get_caller_context(operator_info.name)
            ctx.register_operator_call(ia_op_exec_context.operator_name, ia_op_exec_context.scope_in_model)

            op_input = OperatorInput(list(args), kwargs)
            processed_input = ctx.execute_pre_hooks(ia_op_exec_context, op_input)
            args = tuple(processed_input.op_args)
            kwargs = processed_input.op_kwargs
            fargs = flatten_args(args, kwargs)

            node = ctx.find_operator_node(fargs, ia_op_exec_context)
            if is_debug():
                ctx.register_node_call(ctx.graph.get_node_key_by_id(node.node_id))

            result = operator(*args, **kwargs)

            result = trace_tensors(result, node)
            result = ctx.execute_post_hooks(ia_op_exec_context, result)

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
        if type(self).__name__ in ITERATION_MODULES.registry_dict.keys():
            ctx.reset_operator_call_count_in_scope(ctx.scope)
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

    # pylint:disable=unused-import,redefined-outer-name,reimported

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
    PatchedOperatorInfo("conv2d"),
    PatchedOperatorInfo("conv_transpose2d"),
    PatchedOperatorInfo("max_pool2d"),
    PatchedOperatorInfo("linear"),
    PatchedOperatorInfo("dropout"),
    PatchedOperatorInfo("threshold"),
    PatchedOperatorInfo("batch_norm"),
    PatchedOperatorInfo("avg_pool2d"),
    PatchedOperatorInfo("adaptive_avg_pool2d"),
    PatchedOperatorInfo("sigmoid"),
    PatchedOperatorInfo("softmax"),
    PatchedOperatorInfo("hardtanh"),
    PatchedOperatorInfo("tanh"),
    PatchedOperatorInfo("elu"),
    PatchedOperatorInfo("elu_"),
    PatchedOperatorInfo("prelu"),
    PatchedOperatorInfo("conv3d"),
    PatchedOperatorInfo("conv_transpose3d"),
    PatchedOperatorInfo("max_pool3d"),
    PatchedOperatorInfo("adaptive_max_pool3d"),
    PatchedOperatorInfo("avg_pool3d"),
    PatchedOperatorInfo("adaptive_avg_pool3d"),
    PatchedOperatorInfo("max_unpool3d"),
    PatchedOperatorInfo("dropout3d"),
    PatchedOperatorInfo("pad", ForwardTraceOnly()),
    PatchedOperatorInfo("layer_norm"),
    PatchedOperatorInfo("gelu"),
    PatchedOperatorInfo("embedding")
]

CORE_OPERATORS = [
    PatchedOperatorInfo("cat"),
    PatchedOperatorInfo("relu"),
    PatchedOperatorInfo("relu_"),
    PatchedOperatorInfo("max"),
    PatchedOperatorInfo("min"),
    PatchedOperatorInfo("sigmoid"),
    PatchedOperatorInfo("flatten", ForwardTraceOnly()),
    PatchedOperatorInfo("div"),
    PatchedOperatorInfo("exp"),
    PatchedOperatorInfo("bmm"),
    PatchedOperatorInfo("tanh"),
    PatchedOperatorInfo("erf"),
    PatchedOperatorInfo("matmul"),
    PatchedOperatorInfo("arange"),
    PatchedOperatorInfo("squeeze", ForwardTraceOnly()),
    PatchedOperatorInfo("unsqueeze", ForwardTraceOnly()),
    PatchedOperatorInfo("transpose", ForwardTraceOnly()),
    PatchedOperatorInfo("index_select", ForwardTraceOnly()),
]

TENSOR_OPERATORS = [
    PatchedOperatorInfo("view", ForwardTraceOnly()),
    PatchedOperatorInfo("permute", ForwardTraceOnly()),
    PatchedOperatorInfo("contiguous", ForwardTraceOnly()),
    PatchedOperatorInfo("reshape", ForwardTraceOnly()),
    PatchedOperatorInfo("mean"),
    PatchedOperatorInfo("__iadd__"),
    PatchedOperatorInfo("__add__"),
    PatchedOperatorInfo("__imul__"),
    PatchedOperatorInfo("__mul__"),
    PatchedOperatorInfo("__idiv__"),
    PatchedOperatorInfo("__div__"),
    PatchedOperatorInfo("__truediv__"),
    PatchedOperatorInfo("__getitem__", ForwardTraceOnly()),
    PatchedOperatorInfo("round"),
    PatchedOperatorInfo("squeeze", ForwardTraceOnly()),
    PatchedOperatorInfo("unsqueeze", ForwardTraceOnly()),
    PatchedOperatorInfo("flatten", ForwardTraceOnly()),
    PatchedOperatorInfo("transpose", ForwardTraceOnly()),
    PatchedOperatorInfo("chunk", ForwardTraceOnly()),
    PatchedOperatorInfo("__radd__"),
    PatchedOperatorInfo("masked_fill"),
    PatchedOperatorInfo("matmul"),
    PatchedOperatorInfo("expand", ForwardTraceOnly()),
    PatchedOperatorInfo("index_select", ForwardTraceOnly()),
    PatchedOperatorInfo("masked_fill_", ForwardTraceOnly()),
]


class FunctionQuantizationInfo:
    def __init__(self, name: str, positions_of_args_to_quantize: list):
        self.name = name
        self.positions_of_args_to_quantize = positions_of_args_to_quantize


FUNCTIONS_TO_QUANTIZE = [
    FunctionQuantizationInfo('linear', [0, 1])
]


def get_arg_positions_to_quantize(op_name: str):
    return next((x.positions_of_args_to_quantize for x in FUNCTIONS_TO_QUANTIZE
                 if x.name == op_name), None)


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
    for op_info in PATCHED_OPERATORS:
        op_name = op_info.name
        if hasattr(F, op_name):
            orig = getattr(F, op_name)
            ORIGINAL_OPERATORS[op_name] = orig
            setattr(F, op_info.name, wrap_operator(orig, op_info))
        else:
            warnings.warn("Not patching {} in torch.nn.functional since it is missing in this version of PyTorch"
                          .format(op_name))

    # patch core operators
    import torch
    for op_info in CORE_OPERATORS:
        op_name = op_info.name
        if hasattr(torch, op_name):
            orig = getattr(torch, op_name)
            ORIGINAL_OPERATORS[op_name] = orig
            setattr(torch, op_name, wrap_operator(orig, op_info))
        else:
            warnings.warn("Not patching {} in torch since it is missing in this version of PyTorch"
                          .format(op_name))

    for op_info in TENSOR_OPERATORS:
        op_name = op_info.name
        if hasattr(TracedTensor, op_name):
            orig = getattr(TracedTensor, op_name)
            ORIGINAL_OPERATORS[op_name] = orig
            setattr(TracedTensor, op_name, wrap_operator(orig, op_info))
        else:
            warnings.warn("Not patching {} in torch.Tensor since it is missing in this version of PyTorch"
                          .format(op_name))

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
    for op_info in PATCHED_OPERATORS:
        setattr(F, op_info.name, ORIGINAL_OPERATORS[op_info.name])

    # patch core operators
    import torch
    for op_info in CORE_OPERATORS:
        setattr(torch, op_info.name, ORIGINAL_OPERATORS[op_info.name])

    for op_info in TENSOR_OPERATORS:
        setattr(TracedTensor, op_info.name, ORIGINAL_OPERATORS[op_info.name])

    torch.nn.Module.__call__ = ORIGINAL_OPERATORS["__call__"]
