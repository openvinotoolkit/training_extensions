"""
 Copyright (c) 2020 Intel Corporation
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
import argparse
import cv2 as cv
from utils.ie_tools import load_ie_model


def parse_args():
    parser = argparse.ArgumentParser(description='Eye state classifier')
    parser.add_argument('-m', '--model', required=True, help='Path to model xml file')
    parser.add_argument('-d', '--data_root', required=True, help='Directory with validation images')
    args = parser.parse_args()
    return args


def load_eye_db(root_dir):
    data = []
    for subdir, dirs, files in os.walk(root_dir):
        for i, file in enumerate(files):                
            full_path = os.path.join(subdir, file)
            state = 1 if file[0] == 'o' else 0
            data.append({'filename': full_path, 'label': state})
    return data


def main():
    args = parse_args()

    test_db = load_eye_db(args.data_root)
    net = load_ie_model(args.model, 'CPU', None)
    _, _, height, width = net.get_input_shape().shape

    for sample in test_db:
        img = cv.imread(sample['filename'])
        assert not img is None
        h, w, _ = img.shape
        out = net.forward(cv.resize(img, (width, height)))
        is_open = out[0][0][0][0] < out[0][1][0][0]
        if is_open:
            cv.rectangle(img, (1, 1), (w-1, h-1), (0, 255, 0), 2)
        else:
            cv.rectangle(img, (1, 1), (w-1, h-1), (0, 0, 255), 2)
        cv.imshow("Eye", img)
        cv.waitKey(0)


if __name__ == '__main__':
    main()
