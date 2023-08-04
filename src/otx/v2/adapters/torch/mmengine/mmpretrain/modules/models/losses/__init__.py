"""OTX Algorithms - Classification Losses."""

# Copyright (C) 2023 Intel Corporation
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

from .asymmetric_angular_loss_with_ignore import AsymmetricAngularLossWithIgnore
from .asymmetric_loss_with_ignore import AsymmetricLossWithIgnore
from .barlowtwins_loss import BarlowTwinsLoss
from .cross_entropy_loss import CrossEntropyLossWithIgnore
from .ib_loss import IBLoss

__all__ = [
    "AsymmetricAngularLossWithIgnore",
    "AsymmetricLossWithIgnore",
    "BarlowTwinsLoss",
    "CrossEntropyLossWithIgnore",
    "IBLoss",
]
