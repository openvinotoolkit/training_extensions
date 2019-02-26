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

import os.path as osp
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2 as cv


class NDG(Dataset):
    """NDG Dataset compatible with PyTorch DataLoader."""
    def __init__(self, images_root_path, annotation_list, transform=None, test=False):
        self.test = test
        self.have_landmarks = True
        self.images_root_path = images_root_path
        self.landmarks_file = open(annotation_list, 'r')
        self.samples_info = self._read_samples_info()
        self.transform = transform

    def _read_samples_info(self):
        """Reads annotation of the dataset"""
        samples = []
        data = json.load(self.landmarks_file)

        for image_info in tqdm(data):
            img_name = image_info['path']
            img_path = osp.join(self.images_root_path, img_name)
            landmarks = image_info['lm']
            samples.append((img_path, landmarks))

        return samples

    def __len__(self):
        """Returns total number of samples"""
        return len(self.samples_info)

    def __getitem__(self, idx):
        """Returns sample (image, landmarks) by index"""
        img = cv.imread(self.samples_info[idx][0], cv.IMREAD_COLOR)
        landmarks = self.samples_info[idx][1]
        width, height = img.shape[1], img.shape[0]
        landmarks = np.array([(float(landmarks[i][0]) / width,
                               float(landmarks[i][1]) / height) for i in range(len(landmarks))]).reshape(-1)
        data = {'img': img, 'landmarks': landmarks}
        if self.transform:
            data = self.transform(data)
        return data
