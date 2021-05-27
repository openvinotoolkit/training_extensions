"""
 Copyright (c) 2020-2021 Intel Corporation

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

from .base import ArgConverterMaps
from ..registry import ARG_CONVERTER_MAPS


@ARG_CONVERTER_MAPS.register_module()
class ReidArgConverterMap(ArgConverterMaps):
    @staticmethod
    def _train_compression_base_args_map():
        return {
            'train_data_roots': 'train_data_roots',
            'val_data_roots': 'val_data_roots',
            'train_ann_files': '',
            'val_ann_files': '',
            'resume_from': 'model.resume',
            'load_weights': 'model.load_weights',
            'save_checkpoints_to': 'data.save_dir',
            'classes': 'classes',
            'load_aux_weights': 'load_aux_weights',
        }
    def train_update_args_map(self):
        cur_map = self._train_compression_base_args_map()
        cur_map.update({
            'batch_size': 'train.batch_size',
            'base_learning_rate': 'train.lr',
            'epochs': 'train.max_epoch',
        })
        return cur_map

    def test_update_args_map(self):
        return {
            'test_ann_files': '',
            'test_data_roots': 'test_data_roots',
            'load_weights': 'model.load_weights',
            'classes': 'classes',
            'load_aux_weights': 'load_aux_weights',
        }

    def compress_update_args_map(self):
        return self._train_compression_base_args_map()
