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

import json
import logging
import tempfile

from mmcv.utils import Config

from ote import MMDETECTION_TOOLS

from .base import BaseTrainer
from ..registry import TRAINERS


@TRAINERS.register_module()
class MMDetectionTrainer(BaseTrainer):

    def _get_tools_dir(self):
        return MMDETECTION_TOOLS


@TRAINERS.register_module()
class MMDetectionCustomClassesTrainer(MMDetectionTrainer):

    @staticmethod
    def classes_list_to_update_config_dict(cfg, classes):
        num_classes = len(classes)
        update_config_dict = {
            'data.train.dataset.classes': classes,
            'data.val.classes': classes,
            'data.test.classes': classes,
            'model.bbox_head.num_classes': num_classes
        }
        if hasattr(cfg.model, 'roi_head'):
            if 'mask_head' in cfg.model.roi_head.keys():
                update_config_dict['model.roi_head.mask_head.num_classes'] = num_classes
        return update_config_dict

    def _update_configuration_file(self, config_path, update_config):
        cfg = Config.fromfile(config_path)
        if 'classes' in update_config:
            classes = update_config['classes']
            update_config.pop('classes')
        else:
            annotation_file = cfg.data.train.dataset.ann_file
            ann_file_key = 'data.train.dataset.ann_file'
            if ann_file_key in update_config:
                annotation_file = update_config[ann_file_key]
                annotation_file = annotation_file.split(',')
            if isinstance(annotation_file, (list, tuple)):
                annotation_file = annotation_file[0]
            with open(annotation_file) as read_file:
                categories = sorted(json.load(read_file)['categories'], key=lambda x: x['id'])
            classes = [category_dict['name'] for category_dict in categories]
        update_config_dict = self.classes_list_to_update_config_dict(cfg, classes)
        cfg.merge_from_dict(update_config_dict)
        cfg.dump(config_path) # TODO(lbeynens): check this, it may be dangerous

        return update_config
