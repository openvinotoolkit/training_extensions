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

from ote import MMDETECTION_TOOLS
from ote.utils.misc import run_through_shell
from mmcv.utils import Config
import yaml

from .base import BaseExporter
from ..registry import EXPORTERS
from ..arg_converters.mmdetection import classes_list_to_update_config_dict, load_classes_from_snapshot


@EXPORTERS.register_module()
class MMDetectionExporter(BaseExporter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opset = 11

    def _get_update_config(self, args):
        return ''

    def _export_to_onnx(self, args, tools_dir):
        update_config = self._get_update_config(args)
        run_through_shell(f'python3 {os.path.join(tools_dir, "export.py")} '
                          f'{args["config"]} '
                          f'{args["load_weights"]} '
                          f'{args["save_model_to"]} '
                          f'{update_config} '
                          f'--opset={self.opset} '
                          f'onnx ')

    def _export_to_openvino(self, args, tools_dir):
        update_config = self._get_update_config(args)
        run_through_shell(f'python3 {os.path.join(tools_dir, "export.py")} '
                          f'{args["config"]} '
                          f'{args["load_weights"]} '
                          f'{args["save_model_to"]} '
                          f'{update_config} '
                          f'--opset={self.opset} '
                          f'openvino '
                          f'--input_format {args["openvino_input_format"]}')

        # FIXME(ikrylov): remove alt_ssd_export block as soon as it becomes useless.
        config = Config.fromfile(args["config"])
        should_run_alt_ssd_export = (hasattr(config.model, 'bbox_head')
                                     and config.model.bbox_head.type == 'SSDHead')

        if should_run_alt_ssd_export:
            run_through_shell(f'python3 {os.path.join(tools_dir, "export.py")} '
                              f'{args["config"]} '
                              f'{args["load_weights"]} '
                              f'{os.path.join(args["save_model_to"], "alt_ssd_export")} '
                              f'{update_config} '
                              f'--opset={self.opset} '
                              f'openvino '
                              f'--input_format {args["openvino_input_format"]} '
                              f'--alt_ssd_export ')

    def _get_tools_dir(self):
        return MMDETECTION_TOOLS


@EXPORTERS.register_module()
class MMDetectionCustomClassesExporter(MMDetectionExporter):

    def _get_update_config(self, args):
        classes_from_snapshot = load_classes_from_snapshot(args['load_weights'])
        if not classes_from_snapshot:
            raise RuntimeError(f'There are no CLASSES in meta information of snapshot: {args["load_weights"]}')

        if 'classes' in args and args['classes']:
            classes_from_args = args['classes'].split(',')
            if classes_from_args != classes_from_snapshot:
                raise RuntimeError('Set of classes passed through CLI does not equal to classes stored in snapshot: '
                                   f'{classes_from_args} vs {classes_from_snapshot}')

        update_config_dict = classes_list_to_update_config_dict(args['config'], classes_from_snapshot)
        update_config = '--update_config ' + ' '.join(f'{k}={v}' for k, v in update_config_dict.items())
        update_config = update_config.replace('"', '\\"')

        return update_config

    def _dump_classes(self, args):
        classes = load_classes_from_snapshot(args['load_weights'])
        out_folder = args['save_model_to']
        out_basename = os.path.splitext(args['config'])[0] + '.extra_params.yml'
        with open(os.path.join(out_folder, out_basename), 'w') as write_file:
            yaml.dump({'classes': classes}, write_file)

    def _export_to_onnx(self, args, tools_dir):
        super()._export_to_onnx(args, tools_dir)
        self._dump_classes(args)

    def _export_to_openvino(self, args, tools_dir):
        super()._export_to_openvino(args, tools_dir)
        self._dump_classes(args)
