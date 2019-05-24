import cv2
import os
import os.path as osp
import numpy as np
from random import Random
import skimage
from skimage import transform
import torch
import torch.utils.data as data


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

        data = [torch.from_numpy(lr_image.transpose((2, 0, 1))).float(),
                torch.from_numpy(bic_image.transpose((2, 0, 1))).float()],\
               [torch.from_numpy(hr_image.transpose((2, 0, 1))).float()]

        return data

    def __len__(self):
        return self.count


class DatasetFromSingleImages(data.Dataset):
    def __init__(self, path, patch_size=None, scale=4, aug_resize_factor_range=[0.5, 1.5], count=None,
                 cache_images=False, seed=1337, dataset_size_factor=1):
        super(DatasetFromSingleImages, self).__init__()
        self.path = path
        self.cache_images = cache_images
        self.dataset_size_factor = dataset_size_factor
        self.count = count
        self.aug_resize_factor_range = aug_resize_factor_range

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

        for f in files:
            image = cv2.imread(osp.join(self.path, f))

            if len(image.shape) != 3 or image.shape[2] != 3 or \
                    (self.patch_size is not None and np.any([image.shape[i] * self.aug_resize_factor_range[0] < self.patch_size[i] for i in range(2)])) or \
                    (self.patch_size is None and np.any([image.shape[i] % self.ds_factor for i in range(2)])):
                continue

            if self.cache_images:
                self.cache.append(image)
            self.image_names.append(f)
            cache_count += 1

            if cache_count >= max_count:
                break

        self.count = len(self.image_names)

    def __getitem__(self, index):
        index = index % self.count
        if self.cache_images:
            image = skimage.img_as_float32(self.cache[index])
        else:
            image = skimage.img_as_float32(cv2.imread(osp.join(self.path, self.image_names[index])))

        if self.patch_size is not None:

            h, w, c = image.shape

            resize_rate = self.random.random() * (self.aug_resize_factor_range[1] - self.aug_resize_factor_range[0]) + self.aug_resize_factor_range[0]

            if w == self.patch_size[1]:
                x = 0
            else:
                x = self.random.randint(0, int(w - self.patch_size[1]*resize_rate))

            if h == self.patch_size_ds[0]:
                y = 0
            else:
                y = self.random.randint(0, int(h - self.patch_size[0]*resize_rate))

            sample = transform.resize(image[y:y + self.patch_size[0], x:x + self.patch_size[1], :], output_shape=self.patch_size,
                                      order=3, mode='reflect', anti_aliasing=True, anti_aliasing_sigma=None, preserve_range=True)

            sample_ds = transform.resize(image=sample, output_shape=self.patch_size_ds, order=3, mode='reflect',
                                         anti_aliasing=True, anti_aliasing_sigma=None, preserve_range=True)
            cubic = cv2.resize(sample_ds, sample.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        else:

            h, w, c = image.shape
            if h % self.ds_factor or w % self.ds_factor:
                raise Exception('ERROR: image size should be divisible by scale')

            sample = image
            sample_ds = transform.resize(image=sample, output_shape=[i // self.ds_factor for i in sample.shape[:2]], order=3, mode='reflect',
                                         anti_aliasing=True,
                                         anti_aliasing_sigma=None, preserve_range=True)

            cubic = cv2.resize(sample_ds, sample.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)

        data = [torch.from_numpy(sample_ds.transpose((2, 0, 1))).float(),
                torch.from_numpy(cubic.transpose((2, 0, 1))).float()], \
               [torch.from_numpy(sample.transpose((2, 0, 1))).float()]

        return data

    def __len__(self):
        return self.count*self.dataset_size_factor

