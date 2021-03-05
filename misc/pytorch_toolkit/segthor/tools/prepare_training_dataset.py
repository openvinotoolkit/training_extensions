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

import os
import argparse
import numpy as np
import nibabel as nii
from scipy.ndimage.filters import median_filter
from scipy.ndimage import zoom
import tqdm

from segthor import loader_helper


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch SegTHOR prepare dataset for training")
    parser.add_argument("--input_path", type=str, help="path to train data")
    parser.add_argument("--output_path", type=str, help="path to output dataset")
    parser.add_argument("--new_scale", nargs='+', default=[1.0, 1.0, 2.5], type=float,
                        help="spatial resolution to resample to")
    return parser.parse_args()

def main():
    opt = parse_args()
    print(opt)

    file_list = [f for f in os.listdir(opt.input_path) if os.path.isdir(os.path.join(opt.input_path, f))]
    file_list.sort()
    print(file_list)

    new_scale = np.array(opt.new_scale)
    affine = np.diag(np.append(new_scale, 1))

    for f in tqdm.tqdm(file_list):
        image, gt, header = loader_helper.read_sample(opt.input_path, f, True)

        image = median_filter(image, 3)

        scale = header.header['pixdim'][1:4]

        scale_factor = scale / new_scale

        # resample
        image = zoom(image, scale_factor, order=3, mode='constant', cval=-1024)
        gt = zoom(gt, scale_factor, order=0, mode='constant', cval=0)

        bbox = loader_helper.lung_bbox(image)

        image_crop = image[bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1], bbox[0, 2]:bbox[1, 2]]
        gt_crop = gt[bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1], bbox[0, 2]:bbox[1, 2]]

        os.mkdir(os.path.join(opt.output_path, f))
        new_image = nii.Nifti1Image(image_crop.astype(np.float32), affine)
        new_gt = nii.Nifti1Image(gt_crop.astype(np.float32), affine)

        nii.save(new_image, os.path.join(opt.output_path, f, f + '.nii.gz'))
        nii.save(new_gt, os.path.join(opt.output_path, f, 'GT.nii.gz'))


if __name__ == '__main__':
    main()
