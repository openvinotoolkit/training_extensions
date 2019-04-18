"""
 Copyright (c) 2019 Intel Corporation

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

import os.path as osp

from .coco import COCODataset


def get_dataset(name, with_gt, remove_images_without_gt, transforms=None,
                root_data_dir=osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', 'data')):
    coco_root = osp.join(root_data_dir, 'coco')
    if name == 'coco_2017_train':
        dataset = COCODataset(osp.join(coco_root, 'images', 'train2017'),
                              osp.join(coco_root, 'annotations', 'instances_train2017.json'),
                              with_gt, remove_images_without_gt, transforms)
    elif name == 'coco_2017_val':
        dataset = COCODataset(osp.join(coco_root, 'images', 'val2017'),
                              osp.join(coco_root, 'annotations', 'instances_val2017.json'),
                              with_gt, remove_images_without_gt, transforms)
    else:
        raise ValueError('Invalid dataset identifier "{}".'.format(name))
    return dataset
