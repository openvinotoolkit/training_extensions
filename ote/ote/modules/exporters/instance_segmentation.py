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

import os
import yaml
from math import ceil
from subprocess import run

from ote import MMDETECTION_TOOLS
from mmcv.utils import Config

from .base import BaseExporter
from ..registry import EXPORTERS


@EXPORTERS.register_module()
class InstanceSegmentationExporter(BaseExporter):
    def __init__(self):
        super(InstanceSegmentationExporter, self).__init__()

    def _export_to_openvino(self, args, tools_dir):
        config = Config.fromfile(args["config"])
        height, width = self._get_input_shape(config)
        run(f'python {os.path.join(tools_dir, "export.py")} '
            f'{args["config"]} '
            f'{args["load_weights"]} '
            f'{args["save_model_to"]} '
            f'openvino '
            f'--input_format {args["openvino_input_format"]} '
            f'--input_shape {height} {width}',
            shell=True,
            check=True)

    @staticmethod
    def _get_input_shape(cfg):
        width, height = cfg.data.test.pipeline[1].img_scale
        size_divisor = cfg.data.train.dataset.pipeline[5]['size_divisor']
        width = ceil(width / size_divisor) * size_divisor
        height = ceil(height / size_divisor) * size_divisor
        return height, width

    def _get_tools_dir(self):
        return MMDETECTION_TOOLS
