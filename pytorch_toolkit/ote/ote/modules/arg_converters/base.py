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
from abc import ABCMeta


class BaseArgConverter(metaclass=ABCMeta):
    train_update_args_map = {}
    test_update_args_map = {}
    compress_update_args_map = {}
    train_to_compress_update_args_map = {}
    # TODO(LeonidBeynenson): replace dicts train_update_args_map, test_update_args_map,
    #       and compress_update_args_map with call of a special function
    #       that may be passed to constructor of the class as a
    #       parameter.
    #       This will allow to avoid copying between these dicts.

    train_out_args_map = {
        'gpu_num': 'gpu_num',
        'tensorboard_dir': 'tensorboard_dir'
    }
    compress_out_args_map = {
        'gpu_num': 'gpu_num',
        'tensorboard_dir': 'tensorboard_dir',
        'nncf_quantization': 'nncf_quantization',
        'nncf_pruning': 'nncf_pruning',
        'nncf_sparsity': 'nncf_sparsity',
        'nncf_binarization': 'nncf_binarization',
    }
    test_out_args_map = {
        'load_weights': 'snapshot',
        'save_metrics_to': 'out',
        'save_output_to': 'show_dir',
    }

    def __init__(self):
        pass

    def convert_train_args(self, model_template_path, args):
        update_args = self.__map_args(args, self.train_update_args_map)

        extra_args = self._get_extra_train_args(args)
        update_args.update(extra_args)

        template_folder = os.path.dirname(model_template_path)
        converted_args = {
            'config': os.path.join(template_folder, args['config']),
            'out': os.path.join(args['save_checkpoints_to'], model_template_path),
            'update_config': update_args,
        }
        converted_args.update(self.__map_args(args, self.train_out_args_map))

        return converted_args

    def convert_compress_args(self, model_template_path, args):
        update_args = self.__map_args(args, self.compress_update_args_map)

        # TODO(LeonidBeynenson): think on _get_extra_compress_args
        #       Now _get_extra_train_args is used since it's the same
        extra_args = self._get_extra_train_args(args)
        update_args.update(extra_args)

        template_folder = os.path.dirname(model_template_path)
        converted_args = {
            'config': os.path.join(template_folder, args['config']),
            'out': os.path.join(args['save_checkpoints_to'], model_template_path),
            'update_config': update_args,
        }
        converted_args.update(self.__map_args(args, self.compress_out_args_map))

        return converted_args

    def convert_test_args(self, model_template_path, args):
        update_args = self.__map_args(args, self.test_update_args_map)

        extra_args = self._get_extra_test_args(args)
        update_args.update(extra_args)

        template_folder = os.path.dirname(model_template_path)
        converted_args = {
            'config': os.path.join(template_folder, args['config']),
            'update_config': update_args,
        }
        converted_args.update(self.__map_args(args, self.test_out_args_map))

        return converted_args

    def _get_extra_train_args(self, args):
        return {}

    def _get_extra_test_args(self, args):
        return {}

    @staticmethod
    def __map_args(src_args, args_map):
        return {v: src_args[k] for k, v in args_map.items()}
