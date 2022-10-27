"""Collection of utils for task implementation in Segmentation Task."""

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

import copy
import glob
import logging
import math
import os
from collections import defaultdict
from typing import List, Optional, Sequence, Union

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    DirectoryPathCheck,
    check_input_parameters_type,
)
from otx.api.configuration import ConfigurableEnum


logger = logging.getLogger(__name__)


class Models(ConfigurableEnum):
    """
    This Enum represents the types of models for inference
    """
    Segmentation = 'segmentation'
    BlurSegmentation = 'blur_segmentation'
