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
    def __init__(self, images_folder):
        self.impaths = []
        self.probe_classes = []
        for probe_class in os.listdir(images_folder):
            for path in os.listdir(os.path.join(images_folder, probe_class)):
                full_path = os.path.join(images_folder, probe_class, path)
                self.impaths.append(full_path)
                self.probe_classes.append(probe_class)

    def __iter__(self):
        for impath, probe_class in zip(self.impaths, self.probe_classes):
            image = cv2.imread(impath)
            yield image, probe_class
