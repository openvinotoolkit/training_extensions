# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import os.path as osp
from random import Random, choice
import numpy as np
import cv2
import skimage
from skimage import transform
import torch
import torch.utils.data as data
from PIL import Image as pil_image
from tqdm import tqdm

class DatasetFromPairedImages(data.Dataset):
    def __init__(self, path, suffix_lr, suffix_hr, count=None):
        super(DatasetFromPairedImages, self).__init__()

        self.path = path
        self.suffix_lr = suffix_lr
        self.suffix_hr = suffix_hr

        images_names = os.listdir(self.path)
        self.hr_image_names = [f for f in images_names if f.endswith(self.suffix_hr)]
        self.hr_image_names.sort()

        if count is not None:
            self.count = count
        else:
            self.count = len(self.hr_image_names)

    def __getitem__(self, index):
        hr_name = self.hr_image_names[index]
        name_root = hr_name.split(self.suffix_hr)[0]
        lr_name = osp.join(self.path, name_root + self.suffix_lr)
        hr_name = osp.join(self.path, name_root + self.suffix_hr)

        hr_image = skimage.img_as_float32(cv2.imread(hr_name))
        lr_image = skimage.img_as_float32(cv2.imread(lr_name))
        bic_image = cv2.resize(lr_image, hr_image.shape[:2][::-1], cv2.INTER_CUBIC)

        item = [torch.from_numpy(lr_image.transpose((2, 0, 1))).float(),
                torch.from_numpy(bic_image.transpose((2, 0, 1))).float()],\
               [torch.from_numpy(hr_image.transpose((2, 0, 1))).float()]

        return item

    def __len__(self):
        return self.count


class DatasetFromSingleImages(data.Dataset):
    # pylint: disable=too-many-arguments
    def __init__(self, path, patch_size=None, scale=4, aug_resize_factor_range=None, count=None,
                 cache_images=False, seed=1337, dataset_size_factor=1):
        super(DatasetFromSingleImages, self).__init__()
        self.path = path
        self.cache_images = cache_images
        self.dataset_size_factor = dataset_size_factor
        self.count = count
        self.resize_factor = aug_resize_factor_range

        self.patch_size = patch_size
        if self.patch_size is not None:
            if patch_size[0] % scale or patch_size[1] % scale:
                raise Exception('ERROR: patch_size should be divisible by scale')

            self.patch_size_ds = [i // scale for i in patch_size]

        self.ds_factor = scale

        self.cache = []
        self.image_names = []

        self._load_images()

        self.random = Random()
        self.random.seed(seed)

    def _load_images(self):
        all_files = os.listdir(self.path)
        files = [f for f in all_files if f.endswith(('.bmp', '.png', '.jpg'))]
        files.sort()
        cache_count = 0

        max_count = len(files)
        if self.count is not None:
            max_count = self.count

        for f in tqdm(files):
            image_size = np.array(pil_image.open(osp.join(self.path, f)).size)

            if (self.patch_size is not None and \
                 np.any([image_size[i] * self.resize_factor[0] < self.patch_size[i] for i in range(2)])) or \
               (self.patch_size is None and np.any([image_size[i] % self.ds_factor for i in range(2)])):
                continue

            if self.cache_images:
                image = cv2.imread(osp.join(self.path, f))
                self.cache.append(image)
            self.image_names.append(f)
            cache_count += 1

            if cache_count >= max_count:
                break

        self.count = len(self.image_names)
        num_skipped = len(files) - self.count
        if num_skipped:
            print("[WARNING] Skipped {} images".format(num_skipped))

        assert self.count != 0

    def __getitem__(self, index):
        index = index % self.count
        if self.cache_images:
            image = skimage.img_as_float32(self.cache[index])
        else:
            image = skimage.img_as_float32(cv2.imread(osp.join(self.path, self.image_names[index])))

        if self.patch_size is not None:

            h, w, _ = image.shape

            resize_rate = self.random.random() * (self.resize_factor[1] - self.resize_factor[0]) + self.resize_factor[0]

            if w == self.patch_size[1]:
                x = 0
            else:
                x = self.random.randint(0, int(w - self.patch_size[1]*resize_rate))

            if h == self.patch_size_ds[0]:
                y = 0
            else:
                y = self.random.randint(0, int(h - self.patch_size[0]*resize_rate))

            sample = transform.resize(image[y:y + self.patch_size[0], x:x + self.patch_size[1], :],
                                      output_shape=self.patch_size, order=3, mode='reflect', anti_aliasing=True,
                                      anti_aliasing_sigma=None, preserve_range=True)

            sample_ds = transform.resize(image=sample, output_shape=self.patch_size_ds, order=3, mode='reflect',
                                         anti_aliasing=True, anti_aliasing_sigma=None, preserve_range=True)
            cubic = cv2.resize(sample_ds, sample.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        else:

            h, w, _ = image.shape
            if h % self.ds_factor or w % self.ds_factor:
                raise Exception('ERROR: image size should be divisible by scale')

            sample = image
            sample_ds = transform.resize(image=sample,
                                         output_shape=[i // self.ds_factor for i in sample.shape[:2]],
                                         order=3,
                                         mode='reflect',
                                         anti_aliasing=True,
                                         anti_aliasing_sigma=None, preserve_range=True)

            cubic = cv2.resize(sample_ds, sample.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)

        item = [torch.from_numpy(sample_ds.transpose((2, 0, 1))).float(),
                torch.from_numpy(cubic.transpose((2, 0, 1))).float()], \
               [torch.from_numpy(sample.transpose((2, 0, 1))).float()]

        return item

    def __len__(self):
        return self.count*self.dataset_size_factor


class DatasetTextImages(data.Dataset):
    # pylint: disable=too-many-arguments
    def __init__(self, path, patch_size=None, scale=4, aug_resize_factor_range=None,
                 seed=1337, dataset_size_factor=1, rotate=False):
        super(DatasetTextImages, self).__init__()
        self.path = path
        self.dataset_size_factor = dataset_size_factor
        self.resize_factor = aug_resize_factor_range
        self.patch_size = patch_size
        if self.patch_size is not None:
            if patch_size[0] % scale or patch_size[1] % scale:
                raise Exception('ERROR: patch_size should be divisible by scale')

            self.patch_size_ds = [i // scale for i in patch_size]

        self.ds_factor = scale

        self.image_names = []

        self._load_images()

        self.random = Random()
        self.random.seed(seed)

        self.rotate = rotate
        if self.rotate:
            if self.patch_size is None or self.patch_size[0] != self.patch_size[1]:
                raise Exception('ERROR: Disable rotation or set square patch')

    def _load_images(self):
        all_files = os.listdir(self.path)
        files = [f for f in all_files if f.endswith(('.bmp', '.png', '.jpg'))]
        files.sort()
        files = files[:100]
        for f in tqdm(files):
            pimage = pil_image.open(osp.join(self.path, f))
            image_size = np.array(pimage.size)

            if pimage.mode != 'L':
                # Image should be in gray scale format
                continue
            if self.patch_size is not None and \
                 np.any([image_size[i] * self.resize_factor[0] < self.patch_size[i] for i in range(2)]):
                continue
            if self.patch_size is None and np.any([image_size[i] % self.ds_factor for i in range(2)]):
                print(image_size, [image_size[i] % self.ds_factor for i in range(2)])
                continue

            self.image_names.append(f)


        self.count = len(self.image_names)
        num_skipped = len(files) - self.count
        if num_skipped:
            print("[WARNING] Skipped {} images".format(num_skipped))

        assert self.count != 0

    def __getitem__(self, index):
        index = index % self.count
        # Read image in original format
        image = skimage.img_as_float32(cv2.imread(osp.join(self.path, self.image_names[index]), 0))
        image = image.reshape(image.shape[0], image.shape[1], 1)
        if self.patch_size is not None:

            h, w = image.shape[:2]

            resize_rate = self.random.random() * (self.resize_factor[1] - self.resize_factor[0]) + self.resize_factor[0]

            if self.rotate:
                rotate = choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
                if rotate:
                    image = cv2.rotate(image, rotate)
                    if len(image.shape) == 2:
                        image = image.reshape(image.shape[0], image.shape[1], 1)

            if w == self.patch_size[1]:
                x = 0
            else:
                x = self.random.randint(0, int(w - self.patch_size[1]*resize_rate))

            if h == self.patch_size_ds[0]:
                y = 0
            else:
                y = self.random.randint(0, int(h - self.patch_size[0]*resize_rate))

            sample = transform.resize(image[y:y + self.patch_size[0], x:x + self.patch_size[1], :],
                                      output_shape=self.patch_size, order=3, mode='reflect', anti_aliasing=True,
                                      anti_aliasing_sigma=None, preserve_range=True)

            sample_ds = transform.resize(image=sample, output_shape=self.patch_size_ds, order=3, mode='reflect',
                                         anti_aliasing=True, anti_aliasing_sigma=None, preserve_range=True)
        else:

            h, w, _ = image.shape
            if h % self.ds_factor or w % self.ds_factor:
                raise Exception('ERROR: image size should be divisible by scale')

            sample = image
            sample_ds = transform.resize(image=sample,
                                         output_shape=[i // self.ds_factor for i in sample.shape[:2]],
                                         order=3,
                                         mode='reflect',
                                         anti_aliasing=True,
                                         anti_aliasing_sigma=None, preserve_range=True)

        cubic = cv2.resize(sample_ds, sample.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        cubic = cubic.reshape(cubic.shape[0], cubic.shape[1], 1)

        item = [torch.from_numpy(sample_ds.transpose((2, 0, 1))).float(),
                torch.from_numpy(cubic.transpose((2, 0, 1))).float()], \
               [torch.from_numpy(sample.transpose((2, 0, 1))).float()]

        return item

    def __len__(self):
        return self.count*self.dataset_size_factor
