"""
 Copyright (c) 2021 Intel Corporation

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
from os import path as osp

import cv2 as cv

from ote.interfaces.classification.dataset import IClassificationDataset


class ClassificationImageFolder(IClassificationDataset):
    def __init__(self, data_dir, filter_classes=None):

        ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.gif')
        def is_valid(filename):
            return not filename.startswith('.') and filename.lower().endswith(ALLOWED_EXTS)

        def find_classes(dir, filter_names=None):
            if filter_names:
                classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name in filter_names]
            else:
                classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            return class_to_idx

        self.class_to_idx = find_classes(data_dir, filter_classes)

        self.annotation = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = osp.join(data_dir, target_class)
            if not osp.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = osp.join(root, fname)
                    if is_valid(path):
                        self.annotation.append({'label': class_index, 'path': path})

        if not len(self.annotation):
            print('Failed to locate images in folder ' + data_dir + f' with extensions {ALLOWED_EXTS}')


    def __getitem__(self, idx):
        sample = cv.imread(self.annotation[idx]['path'], cv.IMREAD_COLOR)
        label = self.annotation[idx]['label']
        return {'img': sample, 'label': label}

    def __len__(self):
        return len(self.annotation)

    def get_annotation(self):
        return self.annotation

    def get_classes(self):
        return list(self.class_to_idx.keys())
