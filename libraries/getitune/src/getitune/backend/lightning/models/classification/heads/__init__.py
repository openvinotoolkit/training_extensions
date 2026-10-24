# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Head modules for getitune custom model."""

from .linear_head import LinearClsHead
from .multilabel_cls_head import MultiLabelLinearClsHead, MultiLabelNonLinearClsHead
from .vision_transformer_head import VisionTransformerClsHead

__all__ = [
    "LinearClsHead",
    "MultiLabelLinearClsHead",
    "MultiLabelNonLinearClsHead",
    "VisionTransformerClsHead",
]
