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
from abc import ABCMeta, abstractmethod

from ote.utils.misc import run_through_shell


class BaseExporter(metaclass=ABCMeta):
    def __init__(self):
        pass

    def __call__(self, args):
        tools_dir = self._get_tools_dir()

        if args['openvino']:
            self._export_to_openvino(args, tools_dir)

        if args['onnx']:
            self._export_to_onnx(args, tools_dir)

    def _export_to_openvino(self, args, tools_dir):
        run_through_shell(f'python3 {os.path.join(tools_dir, "export.py")} '
                          f'{args["config"]} '
                          f'{args["load_weights"]} '
                          f'{args["save_model_to"]} '
                          f'openvino '
                          f'--input_format {args["openvino_input_format"]}')

    def _export_to_onnx(self, args, tools_dir):
        run_through_shell(f'python3 {os.path.join(tools_dir, "export.py")} '
                          f'{args["config"]} '
                          f'{args["load_weights"]} '
                          f'{args["save_model_to"]} '
                          f'onnx ')

    @abstractmethod
    def _get_tools_dir(self):
        pass
