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
    def __init__(self, max_size=None, window_size=None, size=None):
        assert int(max_size is not None) + int(window_size is not None) + int(size is not None) == 1

        self.short_side_max = None
        self.long_side_max = None
        if max_size is not None:
            self.short_side_max, self.long_side_max = max_size

        self.height_max = None
        self.width_max = None
        if window_size is not None:
            self.height_max, self.width_max = window_size

        self.height = None
        self.width = None
        if size is not None:
            self.height, self.width = size

    def get_scale(self, image_size):
        if self.height is not None:
            scale_x, scale_y = self.width / image_size[1], self.height / image_size[0]
        elif self.height_max is not None:
            im_scale = min(self.height_max / image_size[0], self.width_max / image_size[1])
            scale_x, scale_y = im_scale, im_scale
        else:
            im_scale = min(self.short_side_max / min(image_size), self.long_side_max / max(image_size))
            scale_x, scale_y = im_scale, im_scale
        return scale_x, scale_y

    def __call__(self, sample):
        image_size = sample['image'].shape[:2]
        scale_x, scale_y = self.get_scale(image_size)

        # Resize image.
        sample['image'] = cv2.resize(sample['image'], None, fx=scale_x, fy=scale_y)
        h, w = sample['image'].shape[:2]

        # Resize boxes.
        if 'gt_boxes' in sample:
            sample['gt_boxes'] *= [scale_x, scale_y, scale_x, scale_y]
            sample['gt_boxes'] = np.clip(sample['gt_boxes'], 0, [w - 1, h - 1, w - 1, h - 1])

        # Resize masks.
        if 'gt_masks' in sample:
            sample['gt_masks'] = [[np.clip(part * [scale_x, scale_y], 0, [w - 1, h - 1]) for part in obj]
                                  for obj in sample['gt_masks']]

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '['
        if self.height is not None:
            format_string += 'size = [{}, {}]'.format(self.height, self.width)
        elif self.height_max is not None:
            format_string += 'widow_size = [{}, {}]'.format(self.height_max, self.width_max)
        else:
            format_string += 'max_size = [{}, {}]'.format(self.short_side_max, self.long_side_max)
        format_string += ']'
        return format_string


class RandomResize(object):
    def __init__(self, mode, sizes=None, heights=None, widths=None):
        assert mode in {'max_size', 'window_size', 'size'}
        self.mode = mode
        self.sizes = sizes
        self.heights = heights
        self.widths = widths

        self.size = self.sizes[0] if self.sizes is not None else (self.heights[0], self.widths[0])

    def update(self):
        if self.sizes is not None:
            self.size = self.sizes[int(random.random() * len(self.sizes))]
        else:
            height_idx = int(random.random() * len(self.heights))
            width_idx = int(random.random() * len(self.widths))
            self.size = (self.heights[height_idx], self.widths[width_idx])

    def __call__(self, sample):
        self.update()
        resizer = Resize(**{self.mode: self.size})
        return resizer(sample)

    def __repr__(self):
        format_string = self.__class__.__name__ + '['
        format_string += 'mode = {}'.format(self.mode)
        if self.size is not None:
            format_string += ', sizes = {}'.format(str(self.sizes))
        else:
            format_string += ', heights = {}, widths = {}'.format(str(self.heights), str(self.widths))
        format_string += ']'
        return format_string


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        self.do_flip = False

    def update(self):
        self.do_flip = random.random() < self.prob

    def __call__(self, sample):
        self.update()
        if self.do_flip:
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


class FilterByBoxSize(object):
    def __init__(self, min_box_size, max_box_size):
        self.min_box_size = list(min_box_size)
        self.max_box_size = list(max_box_size)

    def __call__(self, sample):
        if 'gt_boxes' in sample:
            boxes = sample['gt_boxes']
            widths = boxes[:, 2] - boxes[:, 0] + 1
            heights = boxes[:, 3] - boxes[:, 1] + 1
            valid_boxes = np.logical_and(np.logical_and(self.min_box_size[0] < widths, widths < self.max_box_size[0]),
                                         np.logical_and(self.min_box_size[1] < heights, heights < self.max_box_size[1]))
            if not valid_boxes.any():
                sample['gt_boxes'] = np.zeros([1, 4], dtype=np.float32)
                if 'gt_classes' in sample:
                    sample['gt_classes'] = np.zeros(1)
                if 'gt_is_ignored' in sample:
                    sample['gt_is_ignored'] = np.ones(1)
                if 'gt_masks' in sample:
                    sample['gt_masks'] = [[np.zeros([4, 2], dtype=np.float32)], ]
            else:
                boxes = boxes[valid_boxes]
                sample['gt_boxes'] = boxes
                if 'gt_classes' in sample:
                    sample['gt_classes'] = sample['gt_classes'][valid_boxes]
                if 'gt_is_ignored' in sample:
                    sample['gt_is_ignored'] = sample['gt_is_ignored'][valid_boxes]
                if 'gt_masks' in sample:
                    masks = sample['gt_masks']
                    sample['gt_masks'] = list(mask for i, mask in enumerate(masks) if valid_boxes[i])
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += '[min = {}, max = {}]'.format(str(self.min_box_size), str(self.max_box_size))
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
            image = torch.stack((image[2], image[1], image[0]), dim=0)
        sample['image'] = normalize(image, self.mean, self.std)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + \
                        '[mean = {}, std = {}, rgb = {}]'.format(str(list(self.mean)),
                                                                 str(list(self.std)),
                                                                 str(self.rgb))
        return format_string
