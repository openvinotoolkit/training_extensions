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

import sys

import cv2
import torch

from .instance_dataset import InstanceDataset


class VideoDataset(InstanceDataset):
    def __init__(self, path, labels, transforms=None):
        super().__init__(with_gt=False)
        self.classes = labels
        self.classes_num = len(labels)

        self.transforms = transforms

        self.video = cv2.VideoCapture(path)
        assert self.video.isOpened()
        # Unfortunately, one frame lag is always there.
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, index):
        status, frame = self.video.read()
        if not status:
            self.video.release()
            raise StopIteration

        processed_image = frame
        if self.transforms is not None:
            processed_image = self.transforms({'image': frame})['image']

        sample = dict(original_image=frame,
                      meta=dict(original_size=frame.shape[:2],
                                processed_size=processed_image.shape[1:3]),
                      im_data=processed_image,
                      im_info=torch.tensor([processed_image.shape[1], processed_image.shape[2], 1.0],
                                           dtype=torch.float32))
        return sample

    def evaluate(self, *args, **kwargs):
        pass
