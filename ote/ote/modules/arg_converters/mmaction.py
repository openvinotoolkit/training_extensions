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
import json

from .base import ArgConverterMaps
from ..registry import ARG_CONVERTER_MAPS


@ARG_CONVERTER_MAPS.register_module()
class MMActionArgConverterMap(ArgConverterMaps):
    @staticmethod
    def _train_compression_base_args_map():
        return {
            'train_ann_files': 'data.train.ann_file',
            'train_data_roots': 'root_dir',
            'val_ann_files': 'data.val.ann_file',
            'val_data_roots': 'root_dir',
            'resume_from': 'resume_from',
            'load_weights': 'load_from',
            'save_checkpoints_to': 'work_dir',
            'batch_size': 'data.videos_per_gpu',
        }

    def compress_update_args_map(self):
        return self._train_compression_base_args_map()

    def train_update_args_map(self):
        cur_map = self._train_compression_base_args_map()
        cur_map.update({
            'base_learning_rate': 'optimizer.lr',
            'epochs': 'total_epochs',
        })
        return cur_map

    def test_update_args_map(self):
        return {
            'test_ann_files': 'data.test.ann_file',
            'test_data_roots': 'root_dir',
        }

    def _get_extra_args(self, args):
        update_config_dict = dict()

        if 'classes' in args and args['classes']:
            update_config_dict['classes'] = args['classes']

        meta_info_filepath = os.path.splitext(args['config'])[0] + '.meta_info.json'
        if os.path.exists(meta_info_filepath):
            with open(meta_info_filepath) as input_meta_stream:
                meta_data = json.load(input_meta_stream)

            assert 'model_classes' in meta_data
            model_classes = {int(k): str(v) for k, v in meta_data['model_classes'].items()}
            str_model_classes = ','.join([model_classes[k] for k in sorted(model_classes)])
            update_config_dict['model_classes'] = str_model_classes

        return update_config_dict

    def get_extra_train_args(self, args):
        return self._get_extra_args(args)

    def get_extra_test_args(self, args):
        return self._get_extra_args(args)

    def get_extra_compress_args(self, args):
        return self._get_extra_args(args)
