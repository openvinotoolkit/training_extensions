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

import os.path as osp

from .coco_text import COCOTextDataset


def root_data_dir():
    """ Returns root data dir. """
    return osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', 'data', 'coco')


def get_dataset(name, with_gt, remove_images_without_gt, transforms=None,
                root_data_dir=root_data_dir(),
                alphabet_decoder=None,
                remove_images_without_text=False):
    """ Returns dataset. """

    dataset = COCOTextDataset(root_data_dir,
                              osp.join(root_data_dir, name),
                              with_gt, remove_images_without_gt, transforms,
                              remove_images_without_text=remove_images_without_text,
                              alphabet_decoder=alphabet_decoder)
    return dataset
