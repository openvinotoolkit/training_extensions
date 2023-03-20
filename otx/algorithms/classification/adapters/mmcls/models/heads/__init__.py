"""OTX Algorithms - Classification Heads."""

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

from .contrastive_head import ConstrastiveHead
from .custom_cls_head import CustomLinearClsHead, CustomNonLinearClsHead
from .custom_hierarchical_linear_cls_head import CustomHierarchicalLinearClsHead
from .custom_hierarchical_non_linear_cls_head import CustomHierarchicalNonLinearClsHead
from .custom_multi_label_linear_cls_head import CustomMultiLabelLinearClsHead
from .custom_multi_label_non_linear_cls_head import CustomMultiLabelNonLinearClsHead
from .non_linear_cls_head import NonLinearClsHead
from .semisl_cls_head import SemiLinearClsHead, SemiNonLinearClsHead
from .supcon_cls_head import SupConClsHead
from .mmov_cls_head import MMOVClsHead
from .conv_head import ConvClsHead
from .cls_head import ClsHead


__all__ = [
    "ConstrastiveHead",
    "CustomLinearClsHead",
    "CustomNonLinearClsHead",
    "CustomHierarchicalLinearClsHead",
    "CustomHierarchicalNonLinearClsHead",
    "CustomMultiLabelLinearClsHead",
    "CustomMultiLabelNonLinearClsHead",
    "NonLinearClsHead",
    "SemiLinearClsHead",
    "SemiNonLinearClsHead",
    "SupConClsHead",
    "MMOVClsHead",
    "ConvClsHead",
    "ClsHead"
]
