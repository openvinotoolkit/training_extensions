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

from ote import MMACTION_TOOLS
from ote.utils.misc import run_through_shell

from .base import BaseExporter
from ..registry import EXPORTERS


@EXPORTERS.register_module()
class MMActionExporter(BaseExporter):
    def _get_tools_dir(self):
        return MMACTION_TOOLS

    def _get_common_cmd(self, args, tools_dir):
        out_meta_info = os.path.splitext(args["config"])[0] + ".meta_info.json"
        cmd = f'python {os.path.join(tools_dir, "export.py")} '\
              f'{args["config"]} '\
              f'{args["load_weights"]} '\
              f'{args["save_model_to"]} '\
              f'{out_meta_info} '

        update_config = self._get_update_config(args)
        if update_config:
            cmd += update_config

        return cmd

    def _get_update_config(self, args):
        if 'classes' not in args or not args['classes']:
            return None

        classes_from_args = args['classes']
        update_config = f'--update_config classes={classes_from_args}'

        return update_config

    def _export_to_openvino(self, args, tools_dir):
        cmd = self._get_common_cmd(args, tools_dir)
        cmd += f'openvino '\
               f'--input_format {args["openvino_input_format"]} '

        run_through_shell(cmd)

    def _export_to_onnx(self, args, tools_dir):
        cmd = self._get_common_cmd(args, tools_dir)
        cmd += 'onnx '

        run_through_shell(cmd)
