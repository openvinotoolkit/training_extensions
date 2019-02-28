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

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2 as cv


class CelebA(Dataset):
    """CelebA Dataset compatible with PyTorch DataLoader."""
    def __init__(self, images_root_path, landmarks_folder_path, transform=None, test=False):
        self.test = test
        self.have_landmarks = True
        self.images_root_path = images_root_path
        bb_file_name = 'list_bbox_celeba.txt'
        landmarks_file_name = 'list_landmarks_celeba.txt'
        self.detections_file = open(osp.join(landmarks_folder_path, bb_file_name), 'r')
        self.landmarks_file = open(osp.join(landmarks_folder_path, landmarks_file_name), 'r')
        self.samples_info = self._read_samples_info()
        self.transform = transform

    def _read_samples_info(self):
        """Reads annotation of the dataset"""
        samples = []

        detections_file_lines = self.detections_file.readlines()[2:]
        landmarks_file_lines = self.landmarks_file.readlines()[2:]
        assert len(detections_file_lines) == len(landmarks_file_lines)

        if self.test:
            images_range = range(182638, len(landmarks_file_lines))
        else:
            images_range = range(182637)

        for i in tqdm(images_range):
            line = detections_file_lines[i].strip()
            img_name = line.split(' ')[0]
            img_path = osp.join(self.images_root_path, img_name)

            bbox = list(filter(bool, line.split(' ')[1:]))
            bbox = [int(coord) for coord in bbox]
            if bbox[2] == 0 or bbox[3] == 0:
                continue

            line_landmarks = landmarks_file_lines[i].strip().split(' ')[1:]
            landmarks = list(filter(bool, line_landmarks))
            landmarks = [float(coord) for coord in landmarks]
            samples.append((img_path, bbox, landmarks))

        return samples

    def __len__(self):
        """Returns total number of samples"""
        return len(self.samples_info)

    def __getitem__(self, idx):
        """Returns sample (image, landmarks) by index"""
        img = cv.imread(self.samples_info[idx][0], cv.IMREAD_COLOR)
        bbox = self.samples_info[idx][1]
        landmarks = self.samples_info[idx][2]

        img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        landmarks = np.array([(float(landmarks[2*i]-bbox[0]) / bbox[2],
                               float(landmarks[2*i + 1]-bbox[1])/ bbox[3]) \
                               for i in range(len(landmarks)//2)]).reshape(-1)
        data = {'img': img, 'landmarks': landmarks}
        if self.transform:
            data = self.transform(data)
        return data
