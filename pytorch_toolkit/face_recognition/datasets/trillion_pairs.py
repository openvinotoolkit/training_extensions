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

import cv2 as cv
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.face_align import FivePointsAligner


class TrillionPairs(Dataset):
    """TrillionPairs Dataset compatible with PyTorch DataLoader. For details visit http://trillionpairs.deepglint.com/data"""
    def __init__(self, images_root_path, image_list_path, test_mode=False, transform=None):
        self.image_list_path = image_list_path
        self.images_root_path = images_root_path
        self.test_mode = test_mode
        self.identities = {}

        assert osp.isfile(image_list_path)
        self.have_landmarks = True

        self.all_samples_info = self._read_samples_info()
        self.samples_info = self.all_samples_info
        self.transform = transform

    def _read_samples_info(self):
        """Reads annotation of the dataset"""
        samples = []

        with open(self.image_list_path, 'r') as f:
            images_file_lines = f.readlines()

            for i in tqdm(range(len(images_file_lines))):
                line = images_file_lines[i].strip()
                terms = line.split(' ')
                path = terms[0]
                if not self.test_mode:
                    label = int(terms[1])
                    landmarks = terms[2:]
                    if label in self.identities:
                        self.identities[label].append(len(samples))
                    else:
                        self.identities[label] = [len(samples)]
                else:
                    label = 0
                    landmarks = terms[1:]

                landmarks = [float(coord) for coord in landmarks]
                assert(len(landmarks) == 10)
                samples.append((osp.join(self.images_root_path, path).strip(),
                                label, landmarks))

        return samples

    def get_weights(self):
        """Computes weights of the each identity in dataset according to frequency of it's occurance"""
        weights = [0.]*len(self.all_samples_info)
        for i, sample in enumerate(self.all_samples_info):
            weights[i] = float(len(self.all_samples_info)) / len(self.identities[sample[1]])
        return weights

    def get_num_classes(self):
        """Returns total number of identities"""
        return len(self.identities)

    def __len__(self):
        """Returns total number of samples"""
        return len(self.samples_info)

    def __getitem__(self, idx):
        """Returns sample (image, class id, image id) by index"""
        img = cv.imread(self.samples_info[idx][0], cv.IMREAD_COLOR)
        landmarks = self.samples_info[idx][-1]

        img = FivePointsAligner.align(img, landmarks, d_size=(200, 200), normalized=False, show=False)

        if self.transform:
            img = self.transform(img)

        return {'img': img, 'label': self.samples_info[idx][1], 'idx': idx}
