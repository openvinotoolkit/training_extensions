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

import os

import cv2


class FramesProvider:
    def __init__(self, images_list_path):
        self.impaths = []
        self.probe_classes = []

        with open(images_list_path) as f:
            content = [line.strip().split() for line in f.readlines() if line.strip()]

        root = os.path.dirname(images_list_path)

        for impath, label in content:
            self.impaths.append(os.path.join(root, impath))
            self.probe_classes.append(int(label))

    def __iter__(self):
        for impath, probe_class in zip(self.impaths, self.probe_classes):
            image = cv2.imread(impath)
            yield image, probe_class
