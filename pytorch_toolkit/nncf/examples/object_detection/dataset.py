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

from collections import namedtuple

import cv2
import numpy as np
import torch
from examples.object_detection.datasets.coco import COCODataset
from examples.object_detection.datasets.voc0712 import VOCDetection, VOCAnnotationTransform
from examples.object_detection.utils.augmentations import SSDAugmentation
from nncf.dynamic_graph.graph_builder import create_input_infos

VOC_MEAN = (0.406, 0.456, 0.485)
VOC_STD = (0.255, 0.224, 0.229)

Preprocessing = namedtuple('Preprocessing', ('mean', 'std', 'normalize_coef', 'rgb'))


def get_preprocessing(config):
    if 'preprocessing' not in config:
        return Preprocessing(VOC_MEAN, VOC_STD, 255, True)
    preprocessing_config = config.preprocessing
    return Preprocessing(
        preprocessing_config.get('mean', VOC_MEAN),
        preprocessing_config.get('std', VOC_STD),
        preprocessing_config.get('normalize_coef', 255),
        preprocessing_config.get('rgb', True)
    )


def get_training_dataset(dataset_name, path_to_annotations, path_to_imgs, config):
    # for VOC path_to_imgs = path_to_annotations = voc_root
    assert dataset_name in ['voc', 'coco']
    preprocessing = get_preprocessing(config)
    input_info_list = create_input_infos(config)
    image_size = input_info_list[0].shape[-1]
    ssd_transform = SSDAugmentation(
        image_size,
        preprocessing.mean,
        preprocessing.std,
        preprocessing.normalize_coef
    )
    if dataset_name == 'voc':
        training_dataset = VOCDetection(
            path_to_imgs,
            transform=ssd_transform,
            target_transform=VOCAnnotationTransform(keep_difficult=False),
            return_image_info=False,
            rgb=preprocessing.rgb
        )
    if dataset_name == 'coco':
        training_dataset = COCODataset(
            path_to_annotations, path_to_imgs,
            transform=ssd_transform,
            scale_bboxes=True,
            return_image_info=False,
            rgb=preprocessing.rgb
        )

    return training_dataset


def get_testing_dataset(dataset_name, path_to_annotations, path_to_imgs, config):
    # for VOC path_to_imgs = path_to_annotations = voc_root
    assert dataset_name in ['voc', 'coco']
    preprocessing = get_preprocessing(config)
    input_info_list = create_input_infos(config)
    image_size = input_info_list[0].shape[-1]
    transform = BaseTransform(
        image_size,
        preprocessing.mean,
        preprocessing.std,
        preprocessing.normalize_coef
    )
    if dataset_name == 'voc':
        testing_dataset = VOCDetection(
            path_to_imgs, [('2007', 'test')],
            transform=transform,
            target_transform=VOCAnnotationTransform(keep_difficult=True),
            return_image_info=True,
            rgb=preprocessing.rgb
        )
    if dataset_name == 'coco':
        testing_dataset = COCODataset(
            path_to_annotations, path_to_imgs,
            transform=transform,
            scale_bboxes=False,
            return_image_info=True,
            rgb=preprocessing.rgb
        )

    return testing_dataset


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    others = tuple([] for _ in batch[0][2:])
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))

        for o, b in zip(others, sample[2:]):
            o.append(b)

    return (torch.stack(imgs, 0), targets) + others


def base_transform(image, size, mean, std, normalize_coef):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x /= normalize_coef
    x -= mean
    x /= std
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean, std, normalize_coef):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.normalize_coef = normalize_coef

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean, self.std, self.normalize_coef), boxes, labels
