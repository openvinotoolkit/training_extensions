"""Adapters of classification - mmcls."""

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


# The import required to register the backbone used by the OTX Template with the Registry.
import otx.algorithms.common.adapters.mmcv.models as OTXBackbones

from .datasets import OTXClsDataset, SelfSLDataset
from .models import BYOL, ConstrastiveHead, SelfSLMLP
from .optimizer import LARS

# fmt: off
# isort: off
# FIXME: openvino pot library adds stream handlers to root logger
# which makes annoying duplicated logging
from mmcls.utils import get_root_logger  # pylint: disable=wrong-import-order
get_root_logger().propagate = False
# isort:on
# fmt: on

__all__ = ["OTXClsDataset", "SelfSLDataset", "BYOL", "SelfSLMLP", "ConstrastiveHead", "LARS", "OTXBackbones"]
