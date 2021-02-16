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

from .base import BaseArgConverter, ArgConverterMaps
from ..registry import ARG_CONVERTERS


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
            'batch_size': 'train.batch_size',
            'classes': 'classes',
        }
    def train_update_args_map(self):
        cur_map = self._train_compression_base_args_map()
        cur_map.update({
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
        }

    def compress_update_args_map(self):
        return self._train_compression_base_args_map()

    def train_out_args_map(self):
        return super().train_out_args_map()

    def compress_out_args_map(self):
        return super().compress_out_args_map()

    def test_out_args_map(self):
        return super().test_out_args_map()

    def get_extra_train_args(self, args):
        return {}

    def get_extra_test_args(self, args):
        return {}

    def get_extra_compress_args(self, args):
        return {}

@ARG_CONVERTERS.register_module()
class ReidArgsConverter(BaseArgConverter):
    def __init__(self):
        super().__init__(ReidArgConverterMap())
