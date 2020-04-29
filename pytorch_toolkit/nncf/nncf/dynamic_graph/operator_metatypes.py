"""
 Copyright (c) 2020 Intel Corporation
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
from copy import copy
from typing import List, Optional

import torch

from nncf.dynamic_graph.patch_pytorch import CustomTraceFunction, ForwardTraceOnly, MODEL_INPUT_OP_NAME
from nncf.dynamic_graph.version_agnostic_op_names import get_version_agnostic_name
from nncf.hw_config_op_names import HWConfigOpName
from nncf.registry import Registry


class OperatorMetatype:
    """Base class for grouping PyTorch operators based on their semantic meaning.
    Each derived class represents a single semantic group - for example, AddMetatype would
    group together '__iadd__', '__add__' and '__radd__' operations which all define elementwise
    tensor addition.
    Derived classes also specify which PyTorch functions in which modules should be patched
    and in what manner, so that the entire group of operations is visible in the internal graph
    representation. Grouping also allows efficient application of HW specifics to compression of
    certain operation groups.
    """
    name = ""

    # Wrapping specifications for operator calls of the following kind:
    # torch.nn.functional.conv2d
    torch_nn_functional_patch_spec = None  # type: Optional[PatchSpec]

    # Wrapping specifications for operator calls of the following kind:
    # torch.cat
    torch_module_patch_spec = None  # type: Optional[PatchSpec]

    # Wrapping specifications for operator calls of the following kind:
    # x = torch.Tensor(...)
    # x1 = x.view(...)
    torch_tensor_patch_spec = None  # type: Optional[PatchSpec]

    # Names of functions registered as operators via @register_operator to be associated
    # with this metatype
    external_op_names = []  # type: List[str]

    hw_config_names = []  # type: List[HWConfigOpName]

    subtypes = []  # type: List[OperatorSubtype]

    @classmethod
    def get_all_aliases(cls: 'OperatorMetatype') -> List[str]:
        # TODO: disambiguate overlapping function names
        retval = copy(cls.external_op_names)
        if cls.torch_nn_functional_patch_spec is not None:
            for fn_name in cls.torch_nn_functional_patch_spec.underlying_function_names:
                retval.append(fn_name)
        if cls.torch_module_patch_spec is not None:
            for fn_name in cls.torch_module_patch_spec.underlying_function_names:
                retval.append(fn_name)
        if cls.torch_tensor_patch_spec is not None:
            for fn_name in cls.torch_tensor_patch_spec.underlying_function_names:
                retval.append(fn_name)
        return retval

    @classmethod
    def determine_subtype(cls,
                          containing_module: Optional[torch.nn.Module] = None,
                          function_args=None,
                          functions_kwargs=None) -> Optional['OperatorSubtype']:
        matches = []
        for subtype in cls.subtypes:
            if subtype.matches(containing_module,
                               function_args,
                               functions_kwargs):
                matches.append(subtype)
        assert len(matches) <= 1, "Multiple subtypes match operator call " \
                                  "- cannot determine single subtype."
        if not matches:
            return None

        return matches[0]


class PatchSpec:
    def __init__(self,
                 underlying_function_names: List[str],
                 custom_trace_fn: CustomTraceFunction = None):
        """
        :param underlying_function_names: All function names in this list will be wrapped with NNCF
        wrappers that allow corresponding function calls to be registered in NNCF internal graph
        representation of the PyTorch model and to be afterwards considered for compression.
        :param custom_trace_fn: Will be called instead of the regular node search/insertion step
        during the corresponding operator call. Useful to specify this for nodes that have no effect on compression
        and therefore not vital to graph representation, but that should still be accounted for so that the
        graph representation does not become disjoint."""
        self.underlying_function_names = underlying_function_names
        self.custom_trace_fn = custom_trace_fn


class OperatorSubtype(OperatorMetatype):
    """Exact specialization of OperatorMetatype that can only be determined via operator argument
    inspection or owning module attribute inspection, and that may have specialized compression method
    configuration other than the one used for general operations having the type of OperatorMetatype."""

    @classmethod
    def matches(cls, containing_module: Optional[torch.nn.Module] = None,
                function_args=None,
                functions_kwargs=None) -> bool:
        raise NotImplementedError


class OperatorMetatypeRegistry(Registry):
    def __init__(self, name):
        super().__init__(name)
        self._op_name_to_op_meta_dict = {}

    def register(self, name=None):
        name_ = name
        super_register = super()._register

        def wrap(obj: 'OperatorMetatype'):
            cls_name = name_
            if cls_name is None:
                cls_name = obj.__name__
            super_register(obj, cls_name)
            op_names = obj.get_all_aliases()
            for name in op_names:
                name = get_version_agnostic_name(name)
                if name not in self._op_name_to_op_meta_dict:
                    self._op_name_to_op_meta_dict[name] = obj
                else:
                    assert self._op_name_to_op_meta_dict[name] == obj, \
                        "Inconsistent operator metatype registry - single patched op name maps to multiple metatypes!"
            return obj

        return wrap

    def get_operator_metatype_by_op_name(self, op_name: str) -> 'OperatorMetatype':
        return self._op_name_to_op_meta_dict[op_name]


OPERATOR_METATYPES = OperatorMetatypeRegistry("operator_metatypes")


@OPERATOR_METATYPES.register()
class NoopMetatype(OperatorMetatype):
    name = "noop"
    external_op_names = [MODEL_INPUT_OP_NAME]


@OPERATOR_METATYPES.register()
class DepthwiseConv1dSubtype(OperatorSubtype):
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    @classmethod
    def matches(cls, containing_module: Optional[torch.nn.Module] = None,
                function_args=None,
                functions_kwargs=None) -> bool:
        if containing_module.groups == containing_module.in_channels:
            return True
        return False


@OPERATOR_METATYPES.register()
class Conv1dMetatype(OperatorMetatype):
    name = "conv1d"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [DepthwiseConv1dSubtype]
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class DepthwiseConv2dSubtype(OperatorSubtype):
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    @classmethod
    def matches(cls, containing_module: Optional[torch.nn.Module] = None,
                function_args=None,
                functions_kwargs=None) -> bool:
        if containing_module.groups == containing_module.in_channels:
            return True
        return False


@OPERATOR_METATYPES.register()
class Conv2dMetatype(OperatorMetatype):
    name = "conv2d"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    torch_nn_functional_patch_spec = PatchSpec([name])
    subtypes = [DepthwiseConv2dSubtype]


@OPERATOR_METATYPES.register()
class DepthwiseConv3dSubtype(OperatorSubtype):
    hw_config_names = [HWConfigOpName.DEPTHWISECONVOLUTION]

    @classmethod
    def matches(cls, containing_module: Optional[torch.nn.Module] = None,
                function_args=None,
                functions_kwargs=None) -> bool:
        if containing_module.groups == containing_module.in_channels:
            return True
        return False


@OPERATOR_METATYPES.register()
class Conv3dMetatype(OperatorMetatype):
    name = "conv3d"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    subtypes = [DepthwiseConv3dSubtype]
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class ConvTranspose2dMetatype(OperatorMetatype):
    name = "conv_transpose2d"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class ConvTranspose3dMetatype(OperatorMetatype):
    name = "conv_transpose3d"
    hw_config_names = [HWConfigOpName.CONVOLUTION]
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class LinearMetatype(OperatorMetatype):
    name = "linear"
    hw_config_names = [HWConfigOpName.MATMUL]
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class HardTanhMetatype(OperatorMetatype):
    name = "hardtanh"
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class TanhMetatype(OperatorMetatype):
    name = "tanh"
    torch_nn_functional_patch_spec = PatchSpec([name])
    torch_module_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class ELUMetatype(OperatorMetatype):
    name = "elu"
    torch_nn_functional_patch_spec = PatchSpec([name, "elu_"])


@OPERATOR_METATYPES.register()
class PRELUMetatype(OperatorMetatype):
    name = "prelu"
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class LayerNormMetatype(OperatorMetatype):
    name = "layer_norm"
    torch_nn_functional_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.MVN]


@OPERATOR_METATYPES.register()
class GELUMetatype(OperatorMetatype):
    name = "gelu"
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class SigmoidMetatype(OperatorMetatype):
    name = "sigmoid"
    torch_nn_functional_patch_spec = PatchSpec([name])
    torch_module_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class AddMetatype(OperatorMetatype):
    name = "add"
    torch_tensor_patch_spec = PatchSpec(["__add__",
                                         "__iadd__",
                                         "__radd__"])
    hw_config_names = [HWConfigOpName.ADD]


@OPERATOR_METATYPES.register()
class SubMetatype(OperatorMetatype):
    name = "sub"
    torch_tensor_patch_spec = PatchSpec(["__sub__",
                                         "__isub__",
                                         "__rsub__"])
    hw_config_names = [HWConfigOpName.SUBTRACT]


@OPERATOR_METATYPES.register()
class MulMetatype(OperatorMetatype):
    name = "mul"
    torch_tensor_patch_spec = PatchSpec(["__mul__",
                                         "__imul__",
                                         "__rmul__"])
    hw_config_names = [HWConfigOpName.MULTIPLY]


@OPERATOR_METATYPES.register()
class DivMetatype(OperatorMetatype):
    name = "div"
    torch_module_patch_spec = PatchSpec([name])
    torch_tensor_patch_spec = PatchSpec(["__div__",
                                         "__idiv__",
                                         "__truediv__"])
    hw_config_names = [HWConfigOpName.DIVIDE]


@OPERATOR_METATYPES.register()
class ExpMetatype(OperatorMetatype):
    name = "exp"
    torch_module_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class ErfMetatype(OperatorMetatype):
    name = "erf"
    torch_module_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class MatMulMetatype(OperatorMetatype):
    name = "matmul"
    torch_module_patch_spec = PatchSpec([name, "bmm"])
    torch_tensor_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class MeanMetatype(OperatorMetatype):
    name = "mean"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.REDUCEMEAN]


@OPERATOR_METATYPES.register()
class RoundMetatype(OperatorMetatype):
    name = "round"
    torch_tensor_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class DropoutMetatype(OperatorMetatype):
    name = "dropout"
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class ThresholdMetatype(OperatorMetatype):
    name = "threshold"
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class BatchNormMetatype(OperatorMetatype):
    name = "batch_norm"
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class AvgPool2dMetatype(OperatorMetatype):
    name = "avg_pool2d"
    hw_config_names = [HWConfigOpName.AVGPOOL]
    torch_nn_functional_patch_spec = PatchSpec([name, "adaptive_avg_pool2d"])


@OPERATOR_METATYPES.register()
class AvgPool3dMetatype(OperatorMetatype):
    name = "avg_pool3d"
    hw_config_names = [HWConfigOpName.AVGPOOL]
    torch_nn_functional_patch_spec = PatchSpec([name, "adaptive_avg_pool3d"])


@OPERATOR_METATYPES.register()
class MaxPool2dMetatype(OperatorMetatype):
    name = "max_pool2d"
    torch_nn_functional_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.MAXPOOL]


@OPERATOR_METATYPES.register()
class MaxPool3dMetatype(OperatorMetatype):
    name = "max_pool3d"
    torch_nn_functional_patch_spec = PatchSpec([name, "adaptive_max_pool3d"])
    hw_config_names = [HWConfigOpName.MAXPOOL]


@OPERATOR_METATYPES.register()
class MaxUnpool3dMetatype(OperatorMetatype):
    name = "max_unpool3d"
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class PadMetatype(OperatorMetatype):
    name = "pad"
    torch_nn_functional_patch_spec = PatchSpec([name], ForwardTraceOnly())


@OPERATOR_METATYPES.register()
class CatMetatype(OperatorMetatype):
    name = "cat"
    torch_module_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.CONCAT]


@OPERATOR_METATYPES.register()
class RELUMetatype(OperatorMetatype):
    name = "relu"
    torch_module_patch_spec = PatchSpec([name, "relu_"])


@OPERATOR_METATYPES.register()
class MaxMetatype(OperatorMetatype):
    name = "max"
    torch_module_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.MAXIMUM,
                       HWConfigOpName.REDUCEMAX]


@OPERATOR_METATYPES.register()
class MinMetatype(OperatorMetatype):
    name = "min"
    torch_module_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.MINIMUM]


@OPERATOR_METATYPES.register()
class ARangeMetatype(OperatorMetatype):
    name = "arange"
    torch_module_patch_spec = PatchSpec([name], ForwardTraceOnly())


@OPERATOR_METATYPES.register()
class TransposeMetatype(OperatorMetatype):
    name = "transpose"
    torch_module_patch_spec = PatchSpec([name], ForwardTraceOnly())
    torch_tensor_patch_spec = PatchSpec([name, "permute"], ForwardTraceOnly())
    hw_config_names = [HWConfigOpName.TRANSPOSE]


@OPERATOR_METATYPES.register()
class GatherMetatype(OperatorMetatype):
    name = "gather"
    torch_module_patch_spec = PatchSpec(["index_select", ], ForwardTraceOnly())
    torch_tensor_patch_spec = PatchSpec(["index_select", "__getitem__"], ForwardTraceOnly())


@OPERATOR_METATYPES.register()
class ScatterMetatype(OperatorMetatype):
    name = "scatter"
    torch_tensor_patch_spec = PatchSpec(["masked_fill", "masked_fill_"])


@OPERATOR_METATYPES.register()
class ReshapeMetatype(OperatorMetatype):
    name = "reshape"
    torch_module_patch_spec = PatchSpec(["squeeze", "flatten", "unsqueeze"], ForwardTraceOnly())
    torch_tensor_patch_spec = PatchSpec([name, "view", "flatten", "squeeze", "unsqueeze"], ForwardTraceOnly())
    hw_config_names = [HWConfigOpName.RESHAPE,
                       HWConfigOpName.SQUEEZE,
                       HWConfigOpName.UNSQUEEZE,
                       HWConfigOpName.FLATTEN]


@OPERATOR_METATYPES.register()
class ContiguousMetatype(OperatorMetatype):
    name = "contiguous"
    torch_tensor_patch_spec = PatchSpec([name], ForwardTraceOnly())


@OPERATOR_METATYPES.register()
class SplitMetatype(OperatorMetatype):
    name = "split"
    torch_tensor_patch_spec = PatchSpec(["chunk"], ForwardTraceOnly())
    hw_config_names = [HWConfigOpName.SPLIT]


@OPERATOR_METATYPES.register()
class ExpandMetatype(OperatorMetatype):
    name = "expand"
    torch_tensor_patch_spec = PatchSpec([name], ForwardTraceOnly())


# Non-quantizable ops
@OPERATOR_METATYPES.register()
class EmbeddingMetatype(OperatorMetatype):
    name = "embedding"
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class SoftmaxMetatype(OperatorMetatype):
    name = "softmax"
    torch_nn_functional_patch_spec = PatchSpec([name])


@OPERATOR_METATYPES.register()
class LessMetatype(OperatorMetatype):
    name = "__lt__"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.LESS]


@OPERATOR_METATYPES.register()
class LessEqualMetatype(OperatorMetatype):
    name = "__le__"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.LESSEQUAL]


@OPERATOR_METATYPES.register()
class GreaterMetatype(OperatorMetatype):
    name = "__gt__"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.GREATER]


@OPERATOR_METATYPES.register()
class GreaterEqualMetatype(OperatorMetatype):
    name = "__ge__"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.GREATEREQUAL]


@OPERATOR_METATYPES.register()
class ModMetatype(OperatorMetatype):
    name = "__mod__"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.FLOORMOD]


@OPERATOR_METATYPES.register()
class EqualsMetatype(OperatorMetatype):
    name = "__eq__"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.EQUAL]


@OPERATOR_METATYPES.register()
class NotEqualMetatype(OperatorMetatype):
    name = "__ne__"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.NOTEQUAL]


@OPERATOR_METATYPES.register()
class LogicalOrMetatype(OperatorMetatype):
    name = "__or__"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.LOGICALOR]


@OPERATOR_METATYPES.register()
class LogicalXorMetatype(OperatorMetatype):
    name = "__xor__"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.LOGICALXOR]


@OPERATOR_METATYPES.register()
class LogicalAndMetatype(OperatorMetatype):
    name = "__and__"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.LOGICALAND]


@OPERATOR_METATYPES.register()
class LogicalNotMetatype(OperatorMetatype):
    name = "logical_not_"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.LOGICALNOT]


@OPERATOR_METATYPES.register()
class PowerMetatype(OperatorMetatype):
    name = "__pow__"
    torch_tensor_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.POWER]


@OPERATOR_METATYPES.register()
class InterpolateMetatype(OperatorMetatype):
    name = "interpolate"
    torch_nn_functional_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.INTERPOLATE]


@OPERATOR_METATYPES.register()
class RepeatMetatype(OperatorMetatype):
    name = "repeat_interleave"
    torch_module_patch_spec = PatchSpec([name])
    hw_config_names = [HWConfigOpName.TILE]


@OPERATOR_METATYPES.register()
class CloneMetatype(OperatorMetatype):
    name = "clone"
    torch_tensor_patch_spec = PatchSpec([name], ForwardTraceOnly())
