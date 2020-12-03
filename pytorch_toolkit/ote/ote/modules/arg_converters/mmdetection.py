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

class MMDetectionArgConverterMap(ArgConverterMaps):
    @staticmethod
    def _train_compression_base_args_map():
        return {
                'train_ann_files': 'data.train.dataset.ann_file',
                'train_data_roots': 'data.train.dataset.img_prefix',
                'val_ann_files': 'data.val.ann_file',
                'val_data_roots': 'data.val.img_prefix',
                'save_checkpoints_to': 'work_dir',
                'batch_size': 'data.samples_per_gpu',
               }
    @classmethod
    def _train_compression_base_args_map_with_resume_load(cls):
        cur_map = cls._train_compression_base_args_map()
        cur_map.update({
            'resume_from': 'resume_from',
            'load_weights': 'load_from',
            })
        return cur_map

    def train_update_args_map(self):
        cur_map = self._train_compression_base_args_map_with_resume_load()
        cur_map.update({
            'base_learning_rate': 'optimizer.lr',
            'epochs': 'total_epochs',
            })
        return cur_map

    def test_update_args_map(self):
        return {
                'test_ann_files': 'data.test.ann_file',
                'test_data_roots': 'data.test.img_prefix',
               }

    def compress_update_args_map(self):
        return self._train_compression_base_args_map_with_resume_load()

    def train_to_compress_update_args_map(self):
        return self._train_compression_base_args_map()

    def train_out_args_map(self):
        return super().train_out_args_map()

    def compress_out_args_map(self):
        return super().compress_out_args_map()

    def test_out_args_map(self):
        return super().test_out_args_map()

    @staticmethod
    def _get_classes_str_and_num_from_args(args):
        classes = args['classes'].split(',')
        classes_str = '[' + ','.join(f'"{x}"' for x in classes) + ']'
        classes_num = len(classes)
        return classes_str, classes_num

    def get_extra_train_args(self, args):
        if not args.get('classes'):
            return {}

        classes_str, classes_num = self._get_classes_str_and_num_from_args(args)
        return {
                'data.train.dataset.classes': classes_str,
                'data.val.classes': classes_str,
                'model.bbox_head.num_classes': classes_num,
               }


    def get_extra_test_args(self, args):
        if not args.get('classes'):
            return {}

        classes_str, classes_num = self._get_classes_str_and_num_from_args(args)
        return {
                'data.test.classes': classes_str,
                'model.bbox_head.num_classes': classes_num,
               }

@ARG_CONVERTERS.register_module()
class MMDetectionArgsConverter(BaseArgConverter):
    def __init__(self):
        super().__init__(MMDetectionArgConverterMap())
