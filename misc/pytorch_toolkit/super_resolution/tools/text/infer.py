#!/usr/bin/env python3
#
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from argparse import ArgumentParser
import os
import warnings
import cv2
import skimage
import numpy as np
import torch
from torch.autograd import Variable


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', help='Path to checpoint', required=True, type=str)
    parser.add_argument('--output_dir', default=None, help='Output debugirectory')
    parser.add_argument('--threshold', default=125, type=int, help='Threshould for postprocessing')
    parser.add_argument('input_image', help='Image with license plate')
    return parser.parse_args()


def image_to_blob(image):
    blob = image.copy()
    blob = blob.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    blob = skimage.img_as_float32(blob)
    blob = np.array([blob])
    return torch.from_numpy(blob).float().cuda()


def blob_to_img(blob):
    blob = blob.cpu().detach().numpy()
    blob = np.clip(blob, 0.0, 1.0)
    blob = blob.transpose((1, 2, 0))  # Change data layout from CHW to HWC

    # Suppression skimage warning:
    #    UserWarning: Possible precision loss when converting from float32 to uint8
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        blob = skimage.img_as_ubyte(blob)
    return blob


def main():
    args = parse_args()

    # Load model
    model = torch.load(args.model)['model']
    model.eval()

    # Prepare input blobs
    image = cv2.imread(args.input_image, 0)

    assert len(image.shape) == 2

    image = image.reshape(image.shape[0], image.shape[1], 1)
    blob1 = image_to_blob(image)

    # Inference
    result = model([Variable(blob1)])

    # Postprocessing
    out_img = blob_to_img(result[0][0])
    out_img = np.where(out_img > args.threshold, 255, 0)

    outpur_dir = args.output_dir if args.output_dir else os.path.dirname(args.input_image)
    out_path = os.path.join(outpur_dir, 'sr_' + os.path.basename(args.input_image))
    cv2.imwrite(out_path, out_img)
    print('Saved: ', out_path)

if __name__ == '__main__':
    main()
