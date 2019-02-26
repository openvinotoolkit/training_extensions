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
import numpy as np
from torch.utils.data import Dataset

from utils.face_align import FivePointsAligner


class LFW(Dataset):
    """LFW Dataset compatible with PyTorch DataLoader."""
    def __init__(self, images_root_path, pairs_path, landmark_file_path='', transform=None):
        self.pairs_path = pairs_path
        self.images_root_path = images_root_path
        self.landmark_file_path = landmark_file_path
        self.use_landmarks = len(self.landmark_file_path) > 0
        if self.use_landmarks:
            self.landmarks = self._read_landmarks()
        self.pairs = self._read_pairs()
        self.transform = transform

    def _read_landmarks(self):
        """Reads landmarks of the dataset"""
        landmarks = {}
        with open(self.landmark_file_path, 'r') as f:
            for line in f.readlines():
                sp = line.split()
                key = sp[0][sp[0].rfind('/')+1:]
                landmarks[key] = [[int(sp[i]), int(sp[i+1])] for i in range(1, 11, 2)]

        return landmarks

    def _read_pairs(self):
        """Reads annotation of the dataset"""
        pairs = []
        with open(self.pairs_path, 'r') as f:
            for line in f.readlines()[1:]:  # skip header
                pair = line.strip().split()
                pairs.append(pair)

        file_ext = 'jpg'
        lfw_dir = self.images_root_path
        path_list = []

        for pair in pairs:
            if len(pair) == 3:
                path0 = osp.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                id0 = pair[0]
                path1 = osp.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
                id1 = pair[0]
                issame = True
            elif len(pair) == 4:
                path0 = osp.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                id0 = pair[0]
                path1 = osp.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
                id1 = pair[0]
                issame = False

            path_list.append((path0, path1, issame, id0, id1))

        return path_list

    def _load_img(self, img_path):
        """Loads an image from dist, then performs face alignment and applies transform"""
        img = cv.imread(img_path, cv.IMREAD_COLOR)

        if self.use_landmarks:
            landmarks = np.array(self.landmarks[img_path[img_path.rfind('/')+1:]]).reshape(-1)
            img = FivePointsAligner.align(img, landmarks, show=False)

        if self.transform is None:
            return img

        return self.transform(img)

    def show_item(self, index):
        """Saves a pair with a given index to disk"""
        path_1, path_2, _, _, _ = self.pairs[index]
        img1 = cv.imread(path_1)
        img2 = cv.imread(path_2)
        if self.use_landmarks:
            landmarks1 = np.array(self.landmarks[path_1[path_1.rfind('/')+1:]]).reshape(-1)
            landmarks2 = np.array(self.landmarks[path_2[path_2.rfind('/')+1:]]).reshape(-1)
            img1 = FivePointsAligner.align(img1, landmarks1)
            img2 = FivePointsAligner.align(img2, landmarks2)
        else:
            img1 = cv.resize(img1, (400, 400))
            img2 = cv.resize(img2, (400, 400))
        cv.imwrite('misclassified_{}.jpg'.format(index), np.hstack([img1, img2]))

    def __getitem__(self, index):
        """Returns a pair of images and similarity flag by index"""
        (path_1, path_2, is_same, id0, id1) = self.pairs[index]
        img1, img2 = self._load_img(path_1), self._load_img(path_2)

        return {'img1': img1, 'img2': img2, 'is_same': is_same, 'id0': id0, 'id1': id1}

    def __len__(self):
        """Returns total number of samples"""
        return len(self.pairs)
