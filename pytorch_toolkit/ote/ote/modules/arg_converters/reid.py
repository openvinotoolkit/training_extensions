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
class ReidArgsConverter(BaseArgConverter):
    train_update_args_map = {
        'train_data_roots': 'train_data_roots',
        'val_data_roots': 'val_data_roots',
        'train_ann_files': '',
        'val_ann_files': '',
        'resume_from': 'model.resume',
        'load_weights': 'model.load_weights',
        'save_checkpoints_to': 'data.save_dir',
        'batch_size': 'train.batch_size',
        'base_learning_rate': 'train.lr',
        'epochs': 'train.max_epoch',
    }
    test_update_args_map = {
        'test_ann_files': '',
        'test_data_roots': 'test_data_roots',
        'load_weights': 'model.load_weights',
    }

    def __init__(self):
        super(ReidArgsConverter, self).__init__()
