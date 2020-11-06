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

import argparse
import os.path as osp
import shutil

import cv2 as cv
from tqdm import tqdm

from datasets import LccFasdDataset
from demo.demo import FaceDetector


def main():
    """Prepares data for the antispoofing recognition demo"""
    # arguments parcing
    parser = argparse.ArgumentParser(description='prepare LCC FASD')
    parser.add_argument('--fd_model', type=str, required=True, help='path to fd model')
    parser.add_argument('--fd_thresh', type=float, default=0.6, help='Threshold for FD')
    parser.add_argument('--device', type=str, default='CPU')
    parser.add_argument('--root_dir', type=str, required=True, help='LCC FASD root dir')
    parser = argparse.ArgumentParser(description='LCC_FASD')
    args = parser.parse_args()
    face_detector = FaceDetector(args.fd_model, args.fd_thresh, args.device)
    protocols = ['train', 'val', 'test']
    print('===> processing the data...')
    save_dir = osp.abspath(shutil.copytree(args.root_dir, './LCC_FASDcropped',
                           ignore=shutil.ignore_patterns('*.png', '.*')))
    dir_path = osp.abspath(args.root_dir)
    for protocol in protocols:
        data =  LccFasdDataset(root_dir=args.root_dir, protocol=protocol,
                      transform=None, get_img_path=True)
        for image, path in tqdm(data, desc=protocol, total=len(data), leave=False):
            if image.any():
                detection = face_detector.get_detections(image)
                if detection:
                    rect, _ = detection[0]
                    left, top, right, bottom = rect
                    image1=image[top:bottom, left:right]
                    if not image1.any():
                        print(f'bad crop, {path}')
                    else:
                        new_path = path.replace(dir_path, save_dir)
                        cv.imwrite(new_path, image1)
                else:
                    print(f'bad crop, {path}')

if __name__ == "__main__":
    main()
