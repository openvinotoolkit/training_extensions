"""
 Copyright (c) 2020 Intel Corporation

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

from ote import MMACTION_TOOLS
from ote.utils.misc import run_through_shell

from .base import BaseEvaluator
from ..registry import EVALUATORS


@EVALUATORS.register_module()
class MMActionEvaluator(BaseEvaluator):

    def _get_tools_dir(self):
        return MMACTION_TOOLS

    def _get_metric_functions(self):
        from ote.metrics.classification.common import mean_accuracy_eval

        return [mean_accuracy_eval]

    def _get_image_shape(self, cfg):
        image_size = cfg.input_img_size if isinstance(cfg.input_img_size, (tuple, list)) else [cfg.input_img_size] * 2
        assert len(image_size) == 2

        image_shape = [cfg.input_clip_length, image_size[0], image_size[1]]
        image_shape = ' '.join([str(x) for x in image_shape])

        return image_shape

    def _get_complexity_and_size(self, cfg, config_path, work_dir, update_config):
        image_shape = self._get_image_shape(cfg)
        tools_dir = self._get_tools_dir()

        res_complexity = os.path.join(work_dir, 'complexity.json')
        update_config = ' '.join([f'{k}={v}' for k, v in update_config.items()])
        update_config = f' --update_config {update_config}' if update_config else ''
        update_config = update_config.replace('"', '\\"')
        run_through_shell(
            f'python3 {tools_dir}/get_flops.py'
            f' {config_path}'
            f' --shape {image_shape}'
            f' --out {res_complexity}'
            f'{update_config}')

        with open(res_complexity) as read_file:
            content = json.load(read_file)

        return content
