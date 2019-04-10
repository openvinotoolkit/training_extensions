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

import random

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {}'.format(t)
        format_string += '\n)'
        return format_string


class Resize(object):
    def __init__(self, max_size=None, window_size=None):
        assert (max_size is not None) != (window_size is not None)

        self.short_side_max = None
        self.long_side_max = None
        if max_size is not None:
            self.short_side_max, self.long_side_max = max_size

        self.height_max = None
        self.width_max = None
        if window_size is not None:
            self.height_max, self.width_max = window_size

    def get_scale(self, image_size):
        if self.height_max is not None:
            im_scale = min(self.height_max / image_size[0], self.width_max / image_size[1])
        else:
            im_scale = min(self.short_side_max / min(image_size), self.long_side_max / max(image_size))
        return im_scale

    def __call__(self, sample):
        image_size = sample['image'].shape[:2]
        scale = self.get_scale(image_size)

        # Resize image.
        sample['image'] = cv2.resize(sample['image'], None, fx=scale, fy=scale)
        h, w = sample['image'].shape[:2]

        # Resize boxes.
        if 'gt_boxes' in sample:
            sample['gt_boxes'] *= scale
            sample['gt_boxes'] = np.clip(sample['gt_boxes'], 0, [w - 1, h - 1, w - 1, h - 1])

        # Resize masks.
        if 'gt_masks' in sample:
            sample['gt_masks'] = [[np.clip(part * scale, 0, [w - 1, h - 1]) for part in obj]
                                  for obj in sample['gt_masks']]

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '['
        if self.height_max is not None:
            format_string += 'widow_size = [{}, {}]'.format(self.height_max, self.width_max)
        else:
            format_string += 'max_size = [{}, {}]'.format(self.short_side_max, self.long_side_max)
        format_string += ']'
        return format_string


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample['image'] = cv2.flip(sample['image'], 1)
            width = sample['image'].shape[1]

            # Flip boxes.
            if 'gt_boxes' in sample:
                boxes = sample['gt_boxes']
                boxes[:, 0], boxes[:, 2] = width - boxes[:, 2] - 1, width - boxes[:, 0] - 1
                sample['gt_boxes'] = boxes

            # Flip masks.
            if 'gt_masks' in sample:
                polygons = sample['gt_masks']
                for i, obj in enumerate(polygons):
                    for j, part in enumerate(obj):
                        polygons[i][j][:, 0] = width - polygons[i][j][:, 0] - 1
                sample['gt_masks'] = polygons

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '[prob = {}]'.format(self.prob)
        return format_string


class ToTensor(object):
    def __call__(self, sample):
        sample['image'] = torch.as_tensor(sample['image'], dtype=torch.float32).permute(2, 0, 1)
        if 'gt_boxes' in sample:
            sample['gt_boxes'] = torch.as_tensor(sample['gt_boxes'], dtype=torch.float32)
        if 'gt_classes' in sample:
            sample['gt_classes'] = torch.as_tensor(sample['gt_classes'])
        if 'gt_is_ignored' in sample:
            sample['gt_is_ignored'] = torch.as_tensor(sample['gt_is_ignored'])
        # Intentionally, do not convert segmentation polygons to Tensors
        # cause ShallowDataParallel will distribute all Tensors between GPUs and we want
        # these polygons to leave at a CPU side.
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


class Normalize(object):
    def __init__(self, mean, std, rgb=False):
        self.mean = mean
        self.std = std
        self.rgb = rgb

    def __call__(self, sample):
        image = sample['image']
        if self.rgb:
            image = image[::-1]
        sample['image'] = normalize(image, self.mean, self.std)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + \
                        '[mean = {}, std = {}, rgb = {}]'.format(str(list(self.mean)),
                                                                 str(list(self.std)),
                                                                 str(self.rgb))
        return format_string
