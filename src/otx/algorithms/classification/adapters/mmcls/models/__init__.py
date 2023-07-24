"""OTX Algorithms - Classification Models."""

# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .classifiers import BYOL, CustomImageClassifier, SemiSLClassifier, SupConClassifier
from .heads import (
    ClsHead,
    ConstrastiveHead,
    ConvClsHead,
    CustomHierarchicalLinearClsHead,
    CustomHierarchicalNonLinearClsHead,
    CustomLinearClsHead,
    CustomMultiLabelLinearClsHead,
    CustomMultiLabelNonLinearClsHead,
    CustomNonLinearClsHead,
    MMOVClsHead,
    SemiLinearMultilabelClsHead,
    SemiNonLinearMultilabelClsHead,
    SupConClsHead,
)
from .losses import (
    AsymmetricAngularLossWithIgnore,
    AsymmetricLossWithIgnore,
    BarlowTwinsLoss,
    CrossEntropyLossWithIgnore,
    IBLoss,
)
from .necks import SelfSLMLP

__all__ = [
    "BYOL",
    "CustomImageClassifier",
    "SemiSLClassifier",
    "SupConClassifier",
    "CustomLinearClsHead",
    "CustomNonLinearClsHead",
    "CustomMultiLabelNonLinearClsHead",
    "CustomMultiLabelLinearClsHead",
    "CustomHierarchicalLinearClsHead",
    "CustomHierarchicalNonLinearClsHead",
    "AsymmetricAngularLossWithIgnore",
    "SemiLinearMultilabelClsHead",
    "SemiNonLinearMultilabelClsHead",
    "MMOVClsHead",
    "ConvClsHead",
    "ClsHead",
    "AsymmetricLossWithIgnore",
    "BarlowTwinsLoss",
    "IBLoss",
    "CrossEntropyLossWithIgnore",
    "SelfSLMLP",
    "ConstrastiveHead",
    "SupConClsHead",
]
