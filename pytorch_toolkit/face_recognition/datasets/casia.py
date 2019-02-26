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

from tqdm import tqdm
from torch.utils.data import Dataset
import cv2 as cv

from utils.face_align import FivePointsAligner

class CASIA(Dataset):
    """CASIA Dataset compatible with PyTorch DataLoader."""
    def __init__(self, images_root_path, image_list_path, transform, use_landmarks=True):
        self.image_list_path = image_list_path
        self.images_root_path = images_root_path
        self.identities = {}
        self.use_landmarks = use_landmarks
        self.samples_info = self._read_samples_info()
        self.transform = transform

    def _read_samples_info(self):
        """Reads annotation of the dataset"""
        samples = []
        with open(self.image_list_path, 'r') as f:
            for line in tqdm(f.readlines(), 'Preparing CASIA dataset'):
                sample = line.split()
                sample_id = sample[1]
                landmarks = [[sample[i], sample[i+1]] for i in range(2, 12, 2)]
                self.identities[sample_id] = [1]
                samples.append((osp.join(self.images_root_path, sample[0]), sample_id, landmarks))

        return samples

    def get_num_classes(self):
        """Returns total number of identities"""
        return len(self.identities)

    def __len__(self):
        """Returns total number of samples"""
        return len(self.samples_info)

    def __getitem__(self, idx):
        img = cv.imread(self.samples_info[idx][0])
        if self.use_landmarks:
            img = FivePointsAligner.align(img, self.samples_info[idx][2],
                                          d_size=(200, 200), normalized=True, show=False)

        if self.transform:
            img = self.transform(img)
        return {'img': img, 'label': int(self.samples_info[idx][1])}
