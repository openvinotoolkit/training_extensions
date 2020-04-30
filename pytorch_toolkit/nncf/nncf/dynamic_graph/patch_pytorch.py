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
from typing import Callable, List

import warnings
from torch import Tensor
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from nncf.dynamic_graph.trace_tensor import TracedTensor, flatten_args
from nncf.dynamic_graph.wrappers import wrap_operator, wrap_module_call, ignore_scope


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


def register_operator(name=None):
    def wrap(operator):
        op_name = name
        if op_name is None:
            op_name = operator.__name__
        return wrap_operator(operator, PatchedOperatorInfo(op_name))

    return wrap

    # TODO: Use same wrapper for model.forward() calls


def torch_jit_script_wrapper(*args, **kwargs):
    # Torch JIT cannot work with NNCF-modified operators,
    # so at each import of a @torch.jit.script-decorated
    # function we need to un-patch the torch operators
    unpatch_torch_operators()

    # This import statement is required, otherwise we get a
    # "RuntimeError: undefined value torch" inside the real torch.jit.script

    # pylint:disable=unused-import,redefined-outer-name,reimported

    retval = _ORIG_JIT_SCRIPT(*args, **kwargs)
    patch_torch_operators()
    return retval


def get_arg_positions_to_quantize(op_name: str):
    from nncf.dynamic_graph.function_input_quantization import FUNCTIONS_TO_QUANTIZE
    return next((x.positions_of_args_to_quantize for x in FUNCTIONS_TO_QUANTIZE
                 if x.name == op_name), None)


class OriginalOpInfo:
    def __init__(self, name: str, namespace, op):
        self.name = name
        self.namespace = namespace
        self.op = op


ORIGINAL_OPERATORS = []  # type: List[OriginalOpInfo]
_JIT_ALREADY_WRAPPED = False
_OPERATORS_ALREADY_WRAPPED = False
_ORIG_JIT_SCRIPT = None


def patch_torch_jit_script():
    import torch
    orig = getattr(torch.jit, "script")
    ORIGINAL_OPERATORS.append(OriginalOpInfo("script", torch.jit, orig))
    global _ORIG_JIT_SCRIPT
    _ORIG_JIT_SCRIPT = orig
    setattr(torch.jit, "script", torch_jit_script_wrapper)


def patch_namespace_opname(namespace, patched_op_info: PatchedOperatorInfo):
    name = patched_op_info.name
    if hasattr(namespace, name):
        orig = getattr(namespace, name)
        ORIGINAL_OPERATORS.append(OriginalOpInfo(name, namespace, orig))
        setattr(namespace, name, wrap_operator(orig, patched_op_info))
    else:
        warnings.warn("Not patching {} since it is missing in this version of PyTorch"
                      .format(name))


def patch_namespace_by_patchspec(namespace, patchspec: 'PatchSpec'):
    for op_name in patchspec.underlying_function_names:
        patched_op_info = PatchedOperatorInfo(op_name, patchspec.custom_trace_fn)
        patch_namespace_opname(namespace, patched_op_info)


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

    # patch operators
    import torch.nn.functional as F
    import torch
    from nncf.dynamic_graph.operator_metatypes import OPERATOR_METATYPES
    for op_meta_class in OPERATOR_METATYPES.registry_dict.values():  # type: OperatorMetatype
        if op_meta_class.torch_nn_functional_patch_spec is not None:
            ps = op_meta_class.torch_nn_functional_patch_spec
            patch_namespace_by_patchspec(F, ps)
        if op_meta_class.torch_module_patch_spec is not None:
            ps = op_meta_class.torch_module_patch_spec
            patch_namespace_by_patchspec(torch, ps)
        if op_meta_class.torch_tensor_patch_spec is not None:
            ps = op_meta_class.torch_tensor_patch_spec
            patch_namespace_by_patchspec(TracedTensor, ps)

    ORIGINAL_OPERATORS.append(OriginalOpInfo("__call__", torch.nn.Module, torch.nn.Module.__call__))
    torch.nn.Module.__call__ = wrap_module_call(torch.nn.Module.__call__)
    ignore_scope(DataParallel)
    ignore_scope(DistributedDataParallel)


def unpatch_torch_operators():
    global _OPERATORS_ALREADY_WRAPPED
    if not _OPERATORS_ALREADY_WRAPPED:
        return
    _OPERATORS_ALREADY_WRAPPED = False

    for orig_op_info in ORIGINAL_OPERATORS:
        setattr(orig_op_info.namespace, orig_op_info.name, orig_op_info.op)


@register_operator()
def nncf_model_input(tensor: 'torch.Tensor'):
    return tensor


# Access via _original op because by this moment the nncf_model_input name will already be wrapped by wrap_operator
# and its __name__ attribute changed correspondingly.
# pylint:disable=protected-access
MODEL_INPUT_OP_NAME = nncf_model_input._original_op.__name__
