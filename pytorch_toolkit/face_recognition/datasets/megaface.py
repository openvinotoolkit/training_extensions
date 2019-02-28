"""
 Copyright (c) 2018 Intel Corporation
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

import numpy as np
from torch.utils.data import Dataset
import cv2 as cv

from utils.face_align import FivePointsAligner


class MegaFace(Dataset):
    """MegaFace Dataset compatible with PyTorch DataLoader."""
    def __init__(self, images_lsit, transform=None):
        self.samples_info = images_lsit
        self.transform = transform

    def __len__(self):
        """Returns total number of samples"""
        return len(self.samples_info)

    def __getitem__(self, idx):
        """Returns sample (image, index)"""
        img = None
        try:
            img = cv.imread(self.samples_info[idx]['path'], cv.IMREAD_COLOR)
            bbox = self.samples_info[idx]['bbox']
            landmarks = self.samples_info[idx]['landmarks']

            if bbox is not None or landmarks is not None:
                if landmarks is not None:
                    landmarks = np.array(landmarks).reshape(5, -1)
                    landmarks[:,0] = landmarks[:,0]*bbox[2] + bbox[0]
                    landmarks[:,1] = landmarks[:,1]*bbox[3] + bbox[1]
                    img = FivePointsAligner.align(img, landmarks.reshape(-1), d_size=(bbox[2], bbox[3]),
                                                  normalized=False, show=False)
                if bbox is not None and landmarks is None:
                    img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        except BaseException:
            print('Corrupted image!', self.samples_info[idx])
            img = np.zeros((128, 128, 3), dtype='uint8')

        if self.transform:
            img = self.transform(img)

        return {'img': img, 'idx': idx}
