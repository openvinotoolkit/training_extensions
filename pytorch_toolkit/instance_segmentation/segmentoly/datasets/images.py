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

import os
import os.path as osp

import cv2
import torch

from .instance_dataset import InstanceDataset


class ImagesDataset(InstanceDataset):
    def __init__(self, path, labels, extensions=None, skip_non_images=True, transforms=None):
        super().__init__(with_gt=False)
        self.classes = labels
        self.classes_num = len(labels)

        self.transforms = transforms

        self.skip_non_images = skip_non_images
        self.images = []
        if osp.isdir(path):
            self.images = list(osp.join(path, i)
                               for i in sorted(os.listdir(path))
                               if osp.isfile(osp.join(path, i)))
        elif osp.isfile(path):
            self.images = [path, ]
        else:
            raise ValueError('"path" is neither an image file, not a directory with images.')

        if extensions is not None:
            extensions = set(extensions)
            self.images = [i for i in self.images
                           if osp.splitext(i) in extensions]

        self.pos = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = None

        read_one_more_image = True
        while read_one_more_image:
            if self.pos >= len(self.images):
                raise StopIteration
            try:
                image_file_path = self.images[self.pos]
                self.pos += 1
                image = cv2.imread(image_file_path)
                read_one_more_image = False
            except Exception:
                read_one_more_image = self.skip_non_images
                image = None

        if image is None:
            raise StopIteration

        processed_image = image
        if self.transforms is not None:
            processed_image = self.transforms({'image': image})['image']

        sample = dict(original_image=image,
                      meta=dict(original_size=image.shape[:2],
                                processed_size=processed_image.shape[1:3]),
                      im_data=processed_image,
                      im_info=torch.tensor([processed_image.shape[1], processed_image.shape[2], 1.0],
                                           dtype=torch.float32))
        return sample

    def evaluate(self, *args, **kwargs):
        pass
