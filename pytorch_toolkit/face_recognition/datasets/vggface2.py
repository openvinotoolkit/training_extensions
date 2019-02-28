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
import numpy as np
from torch.utils.data import Dataset

from utils.face_align import FivePointsAligner


class VGGFace2(Dataset):
    """VGGFace2 Dataset compatible with PyTorch DataLoader."""
    def __init__(self, images_root_path, image_list_path, landmarks_folder_path='',
                 transform=None, landmarks_training=False):
        self.image_list_path = image_list_path
        self.images_root_path = images_root_path
        self.identities = {}

        self.landmarks_file = None
        self.detections_file = None
        if osp.isdir(landmarks_folder_path):
            if 'train' in image_list_path:
                bb_file_name = 'loose_landmark_train.csv'
                landmarks_file_name = 'loose_bb_train.csv'
            elif 'test' in image_list_path:
                bb_file_name = 'loose_landmark_test.csv'
                landmarks_file_name = 'loose_bb_test.csv'
            else:
                bb_file_name = 'loose_landmark_all.csv'
                landmarks_file_name = 'loose_bb_all.csv'
            self.landmarks_file = open(osp.join(landmarks_folder_path, bb_file_name), 'r')
            self.detections_file = open(osp.join(landmarks_folder_path, landmarks_file_name), 'r')
        self.have_landmarks = not self.landmarks_file is None
        self.landmarks_training = landmarks_training
        if self.landmarks_training:
            assert self.have_landmarks is True

        self.samples_info = self._read_samples_info()

        self.transform = transform

    def _read_samples_info(self):
        """Reads annotation of the dataset"""
        samples = []

        with open(self.image_list_path, 'r') as f:
            last_class_id = -1
            images_file_lines = f.readlines()

            if self.have_landmarks:
                detections_file_lines = self.detections_file.readlines()[1:]
                landmarks_file_lines = self.landmarks_file.readlines()[1:]
                assert len(detections_file_lines) == len(landmarks_file_lines)
                assert len(images_file_lines) == len(detections_file_lines)

            for i in tqdm(range(len(images_file_lines))):
                sample = images_file_lines[i].strip()
                sample_id = int(sample.split('/')[0][1:])
                frame_id = int(sample.split('/')[1].split('_')[0])
                if sample_id in self.identities:
                    self.identities[sample_id].append(len(samples))
                else:
                    last_class_id += 1
                    self.identities[sample_id] = [len(samples)]
                if not self.have_landmarks:
                    samples.append((osp.join(self.images_root_path, sample), last_class_id, frame_id))
                else:
                    _, bbox = detections_file_lines[i].split('",')
                    bbox = [max(int(coord), 0) for coord in bbox.split(',')]
                    _, landmarks = landmarks_file_lines[i].split('",')
                    landmarks = [float(coord) for coord in landmarks.split(',')]
                    samples.append((osp.join(self.images_root_path, sample), last_class_id, sample_id, bbox, landmarks))

        return samples

    def get_weights(self):
        """Computes weights of the each identity in dataset according to frequency of it's occurance"""
        weights = [0.]*len(self.samples_info)
        for i, sample in enumerate(self.samples_info):
            weights[i] = len(self.samples_info) / float(len(self.identities[sample[2]]))

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
        if self.landmarks_training:
            landmarks = self.samples_info[idx][-1]
            bbox = self.samples_info[idx][-2]
            img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            landmarks = [(float(landmarks[2*i]-bbox[0]) / bbox[2],
                          float(landmarks[2*i + 1]-bbox[1])/ bbox[3]) for i in range(len(landmarks)//2)]
            data = {'img': img, 'landmarks': np.array(landmarks)}
            if self.transform:
                data = self.transform(data)
            return data

        if self.have_landmarks:
            landmarks = self.samples_info[idx][-1]
            img = FivePointsAligner.align(img, landmarks, d_size=(200, 200), normalized=False)

        if self.transform:
            img = self.transform(img)

        return {'img': img, 'label': self.samples_info[idx][1], 'instance': self.samples_info[idx][2]}
