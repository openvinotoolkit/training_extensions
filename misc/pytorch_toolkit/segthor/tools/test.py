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
import nibabel as nii
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage.filters import median_filter
from skimage import morphology
import torch
import tqdm

import loader_helper
import train


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch SegTHOR predict")
    parser.add_argument("--name", default="test", type=str, help="Experiment name")
    parser.add_argument("--models_path", default="./models", type=str, help="Path to models folder")
    parser.add_argument("--data_path", default="./data", type=str, help="Path to the data folder")
    parser.add_argument("--output_path", default="./output", type=str, help="Path to the output folder")
    parser.add_argument("--new_scale", nargs='+', default=[1.0, 1.0, 2.5], type=float,
                        help="Spatial resolution to resample to")
    return parser.parse_args()

# pylint: disable=R0914,R0915
def main():
    opt = parse_args()
    print(opt)

    trainer = train.Trainer(name=opt.name, models_root=opt.models_path, rewrite=False, connect_tb=False)
    trainer.load_best()
    trainer.model = trainer.model.module.cpu()
    trainer.state.cuda = False

    files = os.listdir(opt.data_path)
    files = [f for f in files if (f.startswith('Patient') and os.path.isfile(os.path.join(opt.data_path, f)))]
    files.sort()

    for f in tqdm.tqdm(files):
        header = loader_helper.read_nii_header(os.path.join(opt.data_path, f))
        image = np.array(header.get_data()).astype(np.float32)
        original_shape = image.shape

        image = median_filter(image, 3)

        scale = header.header['pixdim'][1:4]

        scale_factor = scale / opt.new_scale
        image = zoom(image, scale_factor, order=3, mode='constant', cval=-1024)

        bbox = loader_helper.lung_bbox(image)

        image_crop = image[bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1], bbox[0, 2]:bbox[1, 2]]

        #=========================================
        new_shape_crop = tuple([loader_helper.closest_to_k(i, 16) for i in image_crop.shape])
        diff = np.array(new_shape_crop) - np.array(image_crop.shape)
        pad_left = diff // 2
        pad_right = diff - pad_left

        new_data_crop = np.pad(image_crop, pad_width=tuple([(pad_left[i], pad_right[i]) for i in range(3)]),
                               mode='reflect')

        std = np.sqrt(loader_helper.mean2 - loader_helper.mean * loader_helper.mean)

        new_data_crop = (new_data_crop - loader_helper.mean) / std

        new_data_crop = torch.from_numpy(new_data_crop[None, None, :, :, :]).float()
        output = trainer.predict([[new_data_crop], ])
        output_crop = output[0].cpu().detach().numpy()[0]

        output_crop = output_crop[:, pad_left[0]:-pad_right[0] or None, pad_left[1]:-pad_right[1] or None,
                                  pad_left[2]:-pad_right[2] or None]

        new_label = np.zeros(shape=(4,)+image.shape)
        new_label[:, bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1], bbox[0, 2]:bbox[1, 2]] = output_crop

        # ==========================

        scale_factor = np.array(original_shape) / np.array(image.shape)
        old_labels = [zoom(new_label[i], scale_factor, order=1, mode='constant', cval=0)[None] for i in range(4)]

        old_label = np.concatenate(tuple(old_labels), axis=0)
        old_label = ((np.argmax(old_label, axis=0) + 1) *
                     np.max((old_label > np.array([0.5, 0.5, 0.5, 0.5]).reshape((-1, 1, 1, 1))).astype(np.int32),
                            axis=0)).astype(np.int32)


        assert old_label.shape == original_shape

        eso_connectivity = morphology.label(old_label == 1)
        heart_connectivity = morphology.label(old_label == 2)
        trachea_connectivity = morphology.label(old_label == 3)
        aorta_connectivity = morphology.label(old_label == 4)
        eso_connectivity = loader_helper.reject_small_regions(eso_connectivity, ratio=0.2)
        heart_connectivity = loader_helper.leave_biggest_region(heart_connectivity)
        trachea_connectivity = loader_helper.leave_biggest_region(trachea_connectivity)
        aorta_connectivity = loader_helper.leave_biggest_region(aorta_connectivity)

        old_label[np.logical_and(old_label == 1, eso_connectivity == 0)] = 0
        old_label[np.logical_and(old_label == 2, heart_connectivity == 0)] = 0
        old_label[np.logical_and(old_label == 3, trachea_connectivity == 0)] = 0
        old_label[np.logical_and(old_label == 4, aorta_connectivity == 0)] = 0


        output_header = nii.Nifti1Image(old_label, header.affine)
        nii.save(output_header, os.path.join(opt.output_path, f[:-7]+'.nii'))

if __name__ == '__main__':
    main()
