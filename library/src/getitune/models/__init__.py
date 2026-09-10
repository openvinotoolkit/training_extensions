# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reimport models from differnt backends for user frendly imports."""

from getitune.backend.huggingface.models import (
    HFDetectionModel,
    HFInstSegModel,
    HFModel,
    HFMulticlassClsModel,
    HFMultilabelClsModel,
    HFSemanticSegModel,
)
from getitune.backend.lightning.models import (
    ATSS,
    DEIMV2,
    RFDETR,
    RTDETR,
    SSD,
    YOLOX,
    DEIMDFine,
    DFine,
    DinoV2Seg,
    EdgeCrafter,
    EfficientNet,
    LiteHRNet,
    MaskRCNN,
    MaskRCNNTV,
    MobileNetV3,
    RFDETRInst,
    RTMDetInst,
    RTMPose,
    SegNext,
    TimmModel,
    TVModel,
    VisionTransformer,
)
from getitune.backend.openvino.models import (
    OVDetectionModel,
    OVInstanceSegmentationModel,
    OVKeypointDetectionModel,
    OVModel,
    OVMulticlassClassificationModel,
    OVMultilabelClassificationModel,
    OVSegmentationModel,
)

try:
    from getitune.backend.ultralytics.models import (
        UltralyticsDetectionModel,
        UltralyticsInstSegModel,
        UltralyticsMultiClassClsModel,
        UltralyticsMultiLabelClsModel,
        UltralyticsSemanticSegModel,
    )
except ImportError:
    UltralyticsDetectionModel = None  # type: ignore[assignment]
    UltralyticsInstSegModel = None  # type: ignore[assignment]
    UltralyticsMultiClassClsModel = None  # type: ignore[assignment]
    UltralyticsMultiLabelClsModel = None  # type: ignore[assignment]
    UltralyticsSemanticSegModel = None  # type: ignore[assignment]

__all__ = [
    "ATSS",
    "DEIMV2",
    "RFDETR",
    "RTDETR",
    "SSD",
    "YOLOX",
    "DEIMDFine",
    "DFine",
    "DinoV2Seg",
    "EdgeCrafter",
    "EfficientNet",
    "HFDetectionModel",
    "HFInstSegModel",
    "HFModel",
    "HFMulticlassClsModel",
    "HFMultilabelClsModel",
    "HFSemanticSegModel",
    "LiteHRNet",
    "MaskRCNN",
    "MaskRCNNTV",
    "MobileNetV3",
    "OVDetectionModel",
    "OVInstanceSegmentationModel",
    "OVKeypointDetectionModel",
    "OVModel",
    "OVMulticlassClassificationModel",
    "OVMultilabelClassificationModel",
    "OVSegmentationModel",
    "RFDETRInst",
    "RTMDetInst",
    "RTMPose",
    "SegNext",
    "TVModel",
    "TimmModel",
    "VisionTransformer",
]

if UltralyticsDetectionModel is not None:
    __all__.extend(
        [
            "UltralyticsDetectionModel",
            "UltralyticsInstSegModel",
            "UltralyticsMultiClassClsModel",
            "UltralyticsMultiLabelClsModel",
            "UltralyticsSemanticSegModel",
        ]
    )
