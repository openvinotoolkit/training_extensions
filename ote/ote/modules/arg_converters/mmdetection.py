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
import os

from mmcv import Config
import yaml

from .base import ArgConverterMaps
from ..registry import ARG_CONVERTER_MAPS


def load_classes_from_snapshot(snapshot):
    import torch
    return torch.load(snapshot)['meta'].get('CLASSES', [])

def classes_list_to_update_config_dict(cfg, classes):
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)
    num_classes = len(classes)
    classes = '[' + ','.join(f'"{x}"' for x in classes) + ']'
    update_config_dict = {
        'data.train.dataset.classes': classes,
        'data.val.classes': classes,
        'data.test.classes': classes
    }
    if hasattr(cfg.model, 'bbox_head'):
        update_config_dict['model.bbox_head.num_classes'] = num_classes
    if hasattr(cfg.model, 'roi_head'):
        if 'mask_head' in cfg.model.roi_head.keys():
            update_config_dict['model.roi_head.mask_head.num_classes'] = num_classes
        if 'bbox_head' in cfg.model.roi_head.keys():
            update_config_dict['model.roi_head.bbox_head.num_classes'] = num_classes
    return update_config_dict


@ARG_CONVERTER_MAPS.register_module()
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

@ARG_CONVERTER_MAPS.register_module()
class MMDetectionCustomClassesArgConverterMap(MMDetectionArgConverterMap):
    @staticmethod
    def _get_classes_from_annotation(annotation_file):
        with open(annotation_file) as read_file:
            categories = sorted(json.load(read_file)['categories'], key=lambda x: x['id'])
        classes_from_annotation = [category_dict['name'] for category_dict in categories]
        return classes_from_annotation

    def get_extra_train_args(self, args):
        classes_from_args = None
        if 'classes' in args and args['classes']:
            classes_from_args = args['classes'].split(',')

        classes_from_annotation = self._get_classes_from_annotation(args['train_ann_files'].split(',')[0])

        if classes_from_args:
            if not set(classes_from_args).issubset(set(classes_from_annotation)):
                raise RuntimeError('Set of classes passed through CLI is not subset of classes in training dataset: '
                                   f'{classes_from_args} vs {classes_from_annotation}')
            classes = classes_from_args
        else:
            classes = classes_from_annotation

        snapshot_path = None
        if args['load_weights']:
            snapshot_path = args['load_weights']
        elif args['resume_from']:
            snapshot_path = args['resume_from']
        if snapshot_path:
            classes_from_snapshot = load_classes_from_snapshot(snapshot_path)
            if classes != classes_from_snapshot:
                logging.warning('Set of classes that will be used in current training does not equal to '
                                f'classes stored in snapshot: {classes} vs {classes_from_snapshot}')

        return classes_list_to_update_config_dict(args['config'], classes)

    def get_extra_test_args(self, args):
        classes_from_args = None
        if 'classes' in args and args['classes']:
            classes_from_args = args['classes'].split(',')

        classes_from_snapshot = None
        if args['load_weights'].endswith('.pth'):
            classes_from_snapshot = load_classes_from_snapshot(args['load_weights'])
        else:
            with open(os.path.splitext(args['load_weights'])[0] + '.extra_params.yml') as read_file:
                classes_from_snapshot = yaml.safe_load(read_file)['classes']

        classes_from_annotation = self._get_classes_from_annotation(args['test_ann_files'].split(',')[0])

        if classes_from_args:
            if not set(classes_from_args).issubset(set(classes_from_annotation)):
                raise RuntimeError('Set of classes passed through CLI is not subset of classes in test dataset: '
                                   f'{classes_from_args} vs {classes_from_annotation}')
            if classes_from_args != classes_from_snapshot:
                raise RuntimeError('Set of classes passed through CLI does not equal to classes stored in snapshot: '
                                   f'{classes_from_args} vs {classes_from_snapshot}')
            classes = classes_from_args
        else:
            if classes_from_annotation != classes_from_snapshot:
                raise RuntimeError('Set of classes obtained from test dataset does not equal to '
                                   f'classes stored in snapshot: {classes_from_annotation} vs {classes_from_snapshot}')
            classes = classes_from_annotation

        return classes_list_to_update_config_dict(args['config'], classes)

    def get_extra_compress_args(self, args):
        return self.get_extra_test_args(args)
