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
import math

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F



def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomScaleAligned:
    def __init__(self, min_scale, max_scale, alignment):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.alignment = alignment

    def __call__(self, image, target):
        w, h = image.size
        scale = random.uniform(self.min_scale, self.max_scale)
        w_aligned = math.ceil(w * scale / self.alignment) * self.alignment
        h_aligned = math.ceil(h * scale / self.alignment) * self.alignment
        image = F.resize(image, (w_aligned, h_aligned))
        target = F.resize(target, (w_aligned, h_aligned), interpolation=Image.NEAREST)
        return image, target


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=0)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class RandomSizedCrop:
    """Note: Preserves image aspect ratio. The resulting crop size will differ from
    the original image size by a random factor in the interval [min_scale; 1.0]."""

    def __init__(self, min_scale):
        self.min_scale = min_scale

    def __call__(self, image, target):
        w, h = image.size
        scale = random.uniform(self.min_scale, 1.0)
        crop_w = math.ceil(w * scale)
        crop_h = math.ceil(h * scale)
        crop_params = T.RandomCrop.get_params(image, (crop_h, crop_w))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
