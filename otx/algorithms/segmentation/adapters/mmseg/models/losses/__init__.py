"""Segmentation losses."""

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

from .cross_entropy_loss_with_ignore import CrossEntropyLossWithIgnore
from .detcon_loss import DetConLoss
from .pixel_prototype_ce_loss import PixelPrototypeCELoss

__all__ = ["DetConLoss", "CrossEntropyLossWithIgnore", "PixelPrototypeCELoss"]
