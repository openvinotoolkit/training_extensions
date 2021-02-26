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
import random
from datetime import datetime
import numpy as np
from scipy.ndimage import affine_transform
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import torch
import torch.utils.data as data

from segthor import loader_helper


#https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
def elastic_transform(image, alpha, sigma, order=3, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * (alpha/2.5)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')

    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))

    return map_coordinates(image, indices, order=order, mode='reflect').reshape(shape)


# pylint: disable=R0914,R0915
class SimpleReader(data.Dataset):
    def __init__(self, path, patch_size, series=None, multiplier=1, patches_from_single_image=1):
        super(SimpleReader, self).__init__()
        self.path = path
        self.patch_size = patch_size
        self.multiplier = multiplier
        self.patches_from_single_image = patches_from_single_image

        if series is None:
            self.series = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        else:
            self.series = series

        self.series.sort()

        self.labels_location = []

        self.__cache()
        self.real_length = len(self.series)

        self.patches_from_current_image = self.patches_from_single_image
        self.current_image_index = 0

        self.__load(self.current_image_index)

    @staticmethod
    def get_data_filename(path, series):
        return os.path.join(path, series, series+'.nii.gz')

    @staticmethod
    def get_label_filename(path, series):
        return os.path.join(path, series, 'GT.nii.gz')

    def __cache(self):
        # cache locations of the labels (bounding boxes) inside the images
        for f in self.series:
            label = loader_helper.read_nii(self.get_label_filename(self.path, f))

            bbox = loader_helper.bbox3(label > 0)

            borders = np.array(label.shape)
            borders_low = np.array(self.patch_size) / 2.0 + 1
            borders_high = borders - np.array(self.patch_size) / 2.0 - 1

            bbox[0] = np.maximum(bbox[0]-100, borders_low)
            bbox[1] = np.minimum(bbox[1]+100, borders_high)

            self.labels_location.append(bbox)


    def __load(self, index):
        if self.patches_from_current_image > self.patches_from_single_image:
            self.patches_from_current_image = 0
            self.current_image_index = index
            filename = self.get_data_filename(self.path, self.series[index])
            labelname = self.get_label_filename(self.path, self.series[index])
            self.image = loader_helper.read_nii(filename)
            self.label = loader_helper.read_nii(labelname)

            std = np.sqrt(loader_helper.mean2 - loader_helper.mean * loader_helper.mean)

            self.image = (self.image - loader_helper.mean) / std

        self.patches_from_current_image += 1

    def __getitem__(self, index):
        index = index % self.real_length
        self.__load(index)
        center = np.random.rand(3)

        bbox = self.labels_location[self.current_image_index]

        center = center * (bbox[1] - bbox[0]) + bbox[0]
        left_bottom = center - np.array(self.patch_size) / 2.0
        left_bottom = left_bottom.astype(np.int32)

        data_out = self.image[left_bottom[0]:left_bottom[0] + self.patch_size[0],
                              left_bottom[1]:left_bottom[1] + self.patch_size[1],
                              left_bottom[2]:left_bottom[2] + self.patch_size[2]]

        label_out = self.label[left_bottom[0]:left_bottom[0] + self.patch_size[0],
                               left_bottom[1]:left_bottom[1] + self.patch_size[1],
                               left_bottom[2]:left_bottom[2] + self.patch_size[2]]

        seed = datetime.now().microsecond
        sigma = random.random()*20 + 10
        alpha = random.random()*4000 + 200

        x_scale = 0.7 + random.random()*0.6
        y_scale = 0.7 + random.random()*0.6

        data_out = affine_transform(data_out, (x_scale, y_scale, 1), order=1)
        data_out = elastic_transform(data_out, alpha, sigma, 1, np.random.RandomState(seed))[None]

        label_out = affine_transform(label_out, (x_scale, y_scale, 1), order=0)
        label_out = elastic_transform(label_out, alpha, sigma, 0, np.random.RandomState(seed))
        label_out = np.eye(5)[label_out.astype(np.int32)].transpose((3, 0, 1, 2))

        if random.random() > 0.5:
            data_out = data_out[:, ::-1, :, :].copy()
            label_out = label_out[:, ::-1, :, :].copy()


        if random.random() > 0.5:
            data_out = data_out[:, :, ::-1, :].copy()
            label_out = label_out[:, :, ::-1, :].copy()

        data_out = data_out * (0.6+random.random()*0.8)
        data_out = data_out + 1.2*(random.random() - 0.5)

        labels_torch = torch.from_numpy(label_out[1:].copy()).float()

        return [torch.from_numpy(data_out).float(), ], \
               [labels_torch, ]


    def __len__(self):
        return self.multiplier*self.real_length


class FullReader(data.Dataset):
    def __init__(self, path, series=None):
        super(FullReader, self).__init__()
        self.path = path

        if series is None:
            self.series = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        else:
            self.series = series

        self.series.sort()

    @staticmethod
    def get_data_filename(path, series):
        return os.path.join(path, series, series+'.nii.gz')

    @staticmethod
    def get_label_filename(path, series):
        return os.path.join(path, series, 'GT.nii.gz')

    def __getitem__(self, index):

        filename = self.get_data_filename(self.path, self.series[index])
        labelname = self.get_label_filename(self.path, self.series[index])

        image = loader_helper.read_nii(filename)
        label = loader_helper.read_nii(labelname)

        old_shape = image.shape
        new_shape = tuple([loader_helper.closest_to_k(i, 32) for i in old_shape])
        new_image = np.full(shape=new_shape, fill_value=-1000., dtype=np.float32)
        new_label = np.zeros(shape=new_shape, dtype=np.float32)

        new_image[:old_shape[0], :old_shape[1], :old_shape[2]] = image
        new_label[:old_shape[0], :old_shape[1], :old_shape[2]] = label

        mean = -303.0502877950004
        mean2 = 289439.0029958802
        std = np.sqrt(mean2 - mean * mean)

        new_image = (new_image - mean) / std

        new_label_out = (np.eye(5)[new_label.astype(np.int32)]).transpose((3, 0, 1, 2))

        labels_torch = torch.from_numpy(new_label_out[1:].copy()).float()


        return [torch.from_numpy(new_image[None, :, :, :]).float(), ], \
               [labels_torch, ]

    def __len__(self):
        return len(self.series)
