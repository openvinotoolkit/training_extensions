"""Task of OTX Detection using mmdetection training backend."""

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

from typing import Any, Dict, List

from mmcv.utils import Config, get_git_hash
from mmrotate import __version__

from otx.algorithms.detection.adapters.mmdet.task import MMDetectionTask


class MMRotateTask(MMDetectionTask):
    def record_info_to_checkpoint_meta(self, cfg: Config, classes: List[str]) -> Dict[str, Any]:
        """Record info to checkpoint meta.

        Args:
            cfg (Config): detection configuration
            classes (list): list of dataset classes

        Returns:
            meta: meta information
        """
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmrotate_version=__version__ + get_git_hash()[:7],
                CLASSES=classes,
            )
