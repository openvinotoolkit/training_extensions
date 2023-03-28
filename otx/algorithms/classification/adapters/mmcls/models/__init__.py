"""OTX Algorithms - Classification Models."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from .classifiers import BYOL, SAMImageClassifier, SemiSLClassifier, SupConClassifier
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
    "SAMImageClassifier",
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
