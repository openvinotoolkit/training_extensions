"""Module of otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from .activations import (
    ClampV0,
    EluV0,
    ExpV0,
    GeluV7,
    HardSigmoidV0,
    HSigmoidV5,
    HSwishV4,
    MishV4,
    PReluV0,
    ReluV0,
    SeluV0,
    SigmoidV0,
    SoftMaxV0,
    SoftMaxV1,
    SwishV4,
    TanhV0,
)
from .arithmetics import AddV1, DivideV1, MultiplyV1, SubtractV1, TanV0
from .builder import OPS, OperationRegistry
from .convolutions import ConvolutionV1, GroupConvolutionV1
from .generation import RangeV4
from .image_processings import InterpolateV4
from .infrastructures import ConstantV0, ParameterV0, ResultV0
from .matmuls import EinsumV7, MatMulV0
from .movements import (
    BroadcastV3,
    ConcatV0,
    GatherV0,
    GatherV1,
    PadV1,
    ScatterNDUpdateV3,
    ScatterUpdateV3,
    ShuffleChannelsV0,
    SplitV1,
    StridedSliceV1,
    TileV0,
    TransposeV1,
    VariadicSplitV1,
)
from .normalizations import (
    MVNV6,
    BatchNormalizationV0,
    LocalResponseNormalizationV0,
    NormalizeL2V0,
)
from .object_detections import (
    DetectionOutputV0,
    PriorBoxClusteredV0,
    PriorBoxV0,
    ProposalV4,
    RegionYoloV0,
    ROIPoolingV0,
)
from .op import Attribute, Operation
from .poolings import AvgPoolV1, MaxPoolV0
from .reductions import ReduceMeanV1, ReduceMinV1, ReduceProdV1, ReduceSumV1
from .shape_manipulations import ReshapeV1, ShapeOfV0, ShapeOfV3, SqueezeV0, UnsqueezeV0
from .sorting_maximization import NonMaxSuppressionV5, NonMaxSuppressionV9, TopKV3
from .type_conversions import ConvertV0

__all__ = [
    "SoftMaxV0",
    "SoftMaxV1",
    "ReluV0",
    "SwishV4",
    "SigmoidV0",
    "ClampV0",
    "PReluV0",
    "TanhV0",
    "EluV0",
    "SeluV0",
    "MishV4",
    "HSwishV4",
    "HSigmoidV5",
    "ExpV0",
    "HardSigmoidV0",
    "GeluV7",
    "MultiplyV1",
    "DivideV1",
    "AddV1",
    "SubtractV1",
    "TanV0",
    "OPS",
    "OperationRegistry",
    "ConvolutionV1",
    "GroupConvolutionV1",
    "RangeV4",
    "InterpolateV4",
    "ParameterV0",
    "ResultV0",
    "ConstantV0",
    "MatMulV0",
    "EinsumV7",
    "PadV1",
    "ConcatV0",
    "TransposeV1",
    "GatherV0",
    "GatherV1",
    "StridedSliceV1",
    "SplitV1",
    "VariadicSplitV1",
    "ShuffleChannelsV0",
    "BroadcastV3",
    "ScatterNDUpdateV3",
    "ScatterUpdateV3",
    "TileV0",
    "BatchNormalizationV0",
    "LocalResponseNormalizationV0",
    "NormalizeL2V0",
    "MVNV6",
    "ProposalV4",
    "ROIPoolingV0",
    "DetectionOutputV0",
    "RegionYoloV0",
    "PriorBoxV0",
    "PriorBoxClusteredV0",
    "Operation",
    "Attribute",
    "MaxPoolV0",
    "AvgPoolV1",
    "ReduceMeanV1",
    "ReduceProdV1",
    "ReduceMinV1",
    "ReduceSumV1",
    "SqueezeV0",
    "UnsqueezeV0",
    "ReshapeV1",
    "ShapeOfV0",
    "ShapeOfV3",
    "TopKV3",
    "NonMaxSuppressionV5",
    "NonMaxSuppressionV9",
    "ConvertV0",
]
