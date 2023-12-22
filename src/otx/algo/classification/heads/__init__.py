# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Head modules for OTX custom model."""

from .custom_multilabel_linear_cls_head import CustomMultiLabelLinearClsHead
from .custom_multilabel_non_linear_cls_head import CustomMultiLabelNonLinearClsHead

__all__ = ["CustomMultiLabelLinearClsHead", "CustomMultiLabelNonLinearClsHead"]
