"""
 Copyright (c) 2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import json
import os

from ote import MMPOSE_TOOLS
from ote.utils.misc import run_through_shell

from .base import BaseEvaluator
from ..registry import EVALUATORS


@EVALUATORS.register_module()
class MMPoseEvaluator(BaseEvaluator):

    def _get_tools_dir(self):
        return MMPOSE_TOOLS

    def _get_metric_functions(self):
        from ote.metrics.pose_estimation.common import coco_ap_eval

        return [coco_ap_eval]

    def _get_image_shape(self, cfg):
        try:
            image_size = cfg['data']['test']['data_cfg']['image_size']
        except KeyError:
            image_size = cfg['image_size']
        return f'{image_size} {image_size}'

    def _get_complexity_and_size(self, cfg, config_path, work_dir, update_config):
        image_shape = self._get_image_shape(cfg)
        tools_dir = self._get_tools_dir()

        res_complexity = os.path.join(work_dir, 'complexity.json')
        update_config = ' '.join([f'{k}={v}' for k, v in update_config.items()])
        update_config = f' --update_config {update_config}' if update_config else ''
        update_config = update_config.replace('"', '\\"')
        run_through_shell(
            f'python3 {tools_dir}/analysis/get_flops.py'
            f' {config_path}'
            f' --shape {image_shape}'
            f' --out {res_complexity}'
            f'{update_config}')

        with open(res_complexity) as read_file:
            content = json.load(read_file)

        return content
