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
import os.path as osp
from subprocess import run, DEVNULL, CalledProcessError

import yaml
from yaml import Loader

from ote import REID_TOOLS

from .base import BaseExporter
from ..registry import EXPORTERS


@EXPORTERS.register_module()
class ReidExporter(BaseExporter):

    def _export_to_openvino(self, args, tools_dir):
        onnx_model_path = self._get_onnx_model_path(args["save_model_to"], args["config"])
        if not os.path.exists(onnx_model_path):
            self._export_to_onnx(args, tools_dir)

        with open(args["config"], 'r') as f:
            config = yaml.load(f, Loader=Loader)
            mean_values = str(config['data']['norm_mean'])[1:-1]
            mean_values = str([float(s)*255 for s in mean_values.split(',')])
            scale_values = str(config['data']['norm_std'])[1:-1]
            scale_values = str([float(s)*255 for s in scale_values.split(',')])

        # read yaml here to ger mean std
        command_line = f'mo.py --input_model="{onnx_model_path}" ' \
                       f'--mean_values="{mean_values}" ' \
                       f'--scale_values="{scale_values}" ' \
                       f'--output_dir="{args["save_model_to"]}" ' \
                        '--reverse_input_channels'

        try:
            run('mo.py -h', stdout=DEVNULL, stderr=DEVNULL, shell=True, check=True)
        except CalledProcessError:
            print('OpenVINO Model Optimizer not found, please source '
                  'openvino/bin/setupvars.sh before running this script.')
            return

        run(command_line, shell=True, check=True)

    @staticmethod
    def _get_onnx_model_path(save_dir, conf_path):
        return osp.join(save_dir, osp.splitext(osp.basename(conf_path))[0] + '.onnx')

    def _export_to_onnx(self, args, tools_dir):

        if not os.path.exists(args["save_model_to"]):
            os.makedirs(args["save_model_to"])

        onnx_model_path = self._get_onnx_model_path(args["save_model_to"], args["config"])

        run(f'python3 {os.path.join(tools_dir, "convert_to_onnx.py")} '
            f' --config-file {args["config"]}'
            f' --output-name {onnx_model_path}'
             ' --disable-dyn-axes'
            f' model.load_weights {args["load_weights"]}',
            shell=True,
            check=True)

    def _get_tools_dir(self):
        return REID_TOOLS
