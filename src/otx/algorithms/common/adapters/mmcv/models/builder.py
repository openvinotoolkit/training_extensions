"""Builder for OTX Common Algorithm. - mmcv.models."""

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

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

BACKBONES = Registry("models", parent=MMCV_MODELS, scope="otx")
TORCHVISION_BACKBONES = Registry("models", parent=MMCV_MODELS, scope="torchvision")
