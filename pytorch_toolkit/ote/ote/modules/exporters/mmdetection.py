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
from subprocess import run

from mmcv.utils import Config

from ote import MMDETECTION_TOOLS
from ote.utils.misc import run_through_shell

from .base import BaseExporter
from ..registry import EXPORTERS


@EXPORTERS.register_module()
class MMDetectionExporter(BaseExporter):
    def __init__(self):
        super(MMDetectionExporter, self).__init__()

    def _export_to_openvino(self, args, tools_dir):
        super()._export_to_openvino(args, tools_dir)

        # FIXME(ikrylov): remove alt_ssd_export block as soon as it becomes useless.
        # (LeonidBeynenson): Please, note that alt_ssd_export appoach may be applied only
        #                    to SSD models only that were not compressed by NNCF.
        config = Config.fromfile(args["config"])
        should_run_alt_ssd_export = (hasattr(config.model, 'bbox_head')
                                     and config.model.bbox_head.type == 'SSDHead'
                                     and not config.get('nncf_config'))
        if should_run_alt_ssd_export:
            run_through_shell(f'python {os.path.join(tools_dir, "export.py")} '
                              f'{args["config"]} '
                              f'{args["load_weights"]} '
                              f'{os.path.join(args["save_model_to"], "alt_ssd_export")} '
                              f'openvino '
                              f'--input_format {args["openvino_input_format"]} '
                              f'--alt_ssd_export ')

    def _get_tools_dir(self):
        return MMDETECTION_TOOLS
