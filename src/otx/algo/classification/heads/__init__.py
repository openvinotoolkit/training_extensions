# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Head modules for OTX custom model."""

from .custom_hlabel_linear_cls_head import CustomHierarchicalLinearClsHead
from .custom_hlabel_non_linear_cls_head import CustomHierarchicalNonLinearClsHead
from .custom_multilabel_linear_cls_head import CustomMultiLabelLinearClsHead
from .custom_multilabel_non_linear_cls_head import CustomMultiLabelNonLinearClsHead
from .hlabel_cls_head import HierarchicalLinearClsHead, HierarchicalNonLinearClsHead
from .linear_head import LinearClsHead
from .multilabel_cls_head import MultiLabelLinearClsHead, MultiLabelNonLinearClsHead
from .vision_transformer_head import VisionTransformerClsHead

__all__ = [
    "CustomMultiLabelLinearClsHead",
    "CustomMultiLabelNonLinearClsHead",
    "CustomHierarchicalLinearClsHead",
    "CustomHierarchicalNonLinearClsHead",
    "LinearClsHead",
    "MultiLabelLinearClsHead",
    "MultiLabelNonLinearClsHead",
    "HierarchicalLinearClsHead",
    "HierarchicalNonLinearClsHead",
    "VisionTransformerClsHead",
]
