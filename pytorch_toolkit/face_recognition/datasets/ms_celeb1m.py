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


class MSCeleb1M(Dataset):
    """MSCeleb1M Dataset compatible with PyTorch DataLoader."""
    def __init__(self, images_root_path, image_list_path, transform=None):
        self.image_list_path = image_list_path
        self.images_root_path = images_root_path
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
            last_class_id = -1

            for i in tqdm(range(len(images_file_lines))):
                line = images_file_lines[i]
                terms = line.split('|')
                if len(terms) < 3:
                    continue # FD has failed on this imsage
                path, landmarks, bbox = terms
                image_id, _ = path.split('/')

                if image_id in self.identities:
                    self.identities[image_id].append(len(samples))
                else:
                    last_class_id += 1
                    self.identities[image_id] = [len(samples)]

                bbox = [max(int(coord), 0) for coord in bbox.strip().split(' ')]
                landmarks = [float(coord) for coord in landmarks.strip().split(' ')]
                assert len(bbox) == 4
                assert len(landmarks) == 10
                samples.append((osp.join(self.images_root_path, path).strip(),
                                last_class_id, image_id, bbox, landmarks))

        return samples

    def get_weights(self):
        """Computes weights of the each identity in dataset according to frequency of it's occurance"""
        weights = [0.]*len(self.all_samples_info)
        for i, sample in enumerate(self.all_samples_info):
            weights[i] = float(len(self.all_samples_info)) / len(self.identities[sample[2]])
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
        bbox = self.samples_info[idx][-2]
        landmarks = self.samples_info[idx][-1]

        img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        img = FivePointsAligner.align(img, landmarks, d_size=(200, 200), normalized=True, show=False)

        if self.transform:
            img = self.transform(img)

        return {'img': img, 'label': self.samples_info[idx][1], 'instance': self.samples_info[idx][2]}
