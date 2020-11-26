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

from .base import BaseArgConverter
from ..registry import ARG_CONVERTERS


@ARG_CONVERTERS.register_module()
class MMDetectionArgsConverter(BaseArgConverter):
    # NB: compress_update_args_map is the same as train_update_args_map,
    #     but without base_learning_rate and epochs
    # TODO(LeonidBeynenson): replace the dicts by a function that returns dicts to avoid copying of code
    compress_update_args_map = {
        'train_ann_files': 'data.train.dataset.ann_file',
        'train_data_roots': 'data.train.dataset.img_prefix',
        'val_ann_files': 'data.val.ann_file',
        'val_data_roots': 'data.val.img_prefix',
        'resume_from': 'resume_from',
        'load_weights': 'load_from',
        'save_checkpoints_to': 'work_dir',
        'batch_size': 'data.samples_per_gpu',
    }
    train_update_args_map = {
        'train_ann_files': 'data.train.dataset.ann_file',
        'train_data_roots': 'data.train.dataset.img_prefix',
        'val_ann_files': 'data.val.ann_file',
        'val_data_roots': 'data.val.img_prefix',
        'resume_from': 'resume_from',
        'load_weights': 'load_from',
        'save_checkpoints_to': 'work_dir',
        'batch_size': 'data.samples_per_gpu',
        'base_learning_rate': 'optimizer.lr',
        'epochs': 'total_epochs',
    }
    test_update_args_map = {
        'test_ann_files': 'data.test.ann_file',
        'test_data_roots': 'data.test.img_prefix',
    }

    def __init__(self):
        super(MMDetectionArgsConverter, self).__init__()

    def _get_extra_train_args(self, args):
        out_args = {}
        if 'classes' in args and args['classes']:
            classes = '[' + ','.join(f'"{x}"' for x in args['classes'].split(',')) + ']'
            out_args['data.train.dataset.classes'] = classes
            out_args['data.val.classes'] = classes
            out_args['model.bbox_head.num_classes'] = len(args['classes'].split(','))

        return out_args

    def _get_extra_test_args(self, args):
        out_args = {}
        if 'classes' in args and args['classes']:
            classes = '[' + ','.join(f'"{x}"' for x in args['classes'].split(',')) + ']'
            out_args['data.test.classes'] = classes
            out_args['model.bbox_head.num_classes'] = len(args['classes'].split(','))

        return out_args
