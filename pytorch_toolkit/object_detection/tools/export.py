# Copyright (C) 2020 Intel Corporation
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

# pylint: disable=W0511

import os
from subprocess import run
import yaml

from mmcv import Config

from ote import MODEL_TEMPLATE_FILENAME
from ote.api import export_args_parser
from oteod import MMDETECTION_TOOLS


args = vars(export_args_parser(MODEL_TEMPLATE_FILENAME).parse_args())

if args['openvino']:
    run(f'python3 {os.path.join(MMDETECTION_TOOLS, "export.py")} '
        f'{args["config"]} '
        f'{args["load_weights"]} '
        f'{args["save_model_to"]} '
        f'openvino '
        f'--input_format {args["openvino_input_format"]}',
        shell=True,
        check=True)

    # FIXME(ikrylov): remove alt_ssd_export block as soon as it becomes useless.
    with open(MODEL_TEMPLATE_FILENAME) as read_file:
        config = Config.fromfile(yaml.load(read_file, yaml.SafeLoader)['config'])
        if hasattr(config.model, 'bbox_head'):
            if config.model.bbox_head.type == 'SSDHead':
                run(f'python3 {os.path.join(MMDETECTION_TOOLS, "export.py")} '
                    f'{args["config"]} '
                    f'{args["load_weights"]} '
                    f'{os.path.join(args["save_model_to"], "alt_ssd_export")} '
                    f'openvino '
                    f'--input_format {args["openvino_input_format"]} '
                    f'--alt_ssd_export ',
                    shell=True,
                    check=True)

if args['onnx']:
    run(f'python3 {os.path.join(MMDETECTION_TOOLS, "export.py")} '
        f'{args["config"]} '
        f'{args["load_weights"]} '
        f'{args["save_model_to"]} '
        f'onnx ',
        shell=True,
        check=True)
