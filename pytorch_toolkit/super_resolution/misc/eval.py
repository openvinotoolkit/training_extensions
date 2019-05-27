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

import argparse
import numpy as np
import math
import os
import os.path as osp
from PIL import Image

parser = argparse.ArgumentParser(description="Calculates average PSNR for all pairs od images (*_hr.png and _sr)")
parser.add_argument("--input_folder_path", default="", type=str, help="path to folder with input images")
parser.add_argument("--hr_suffix_name", default="_hr.png", type=str, help="")
parser.add_argument("--sr_suffix_name", default="_sr_x4.png", type=str, help="")
parser.add_argument("--shave_border", default=4, type=int, help="")

def PSNR(pred, gt, shave_border=0):
    pred = np.asarray(pred).astype(np.float)
    gt = np.asarray(gt).astype(np.float)

    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = (pred - gt) / 255.

    r = imdff[:,:,0]
    g = imdff[:,:,1]
    b = imdff[:,:,2]

    y = (r * 65.738 + g * 129.057 + b * 25.064) / 256

    mse = np.mean(y ** 2)
    if mse == 0:
        return np.Infinity

    return - 10 * math.log10(mse)

def main():
    opt = parser.parse_args()

    hr_suffix_name = opt.hr_suffix_name
    sr_suffix_name = opt.sr_suffix_name
    shave_border = opt.shave_border

    input_folder = opt.input_folder_path
    images_names = os.listdir(input_folder)
    hr_image_names = [f for f in images_names if f.endswith(hr_suffix_name)]
    hr_image_names.sort()

    PSNRs = []
    for i, hr_name in enumerate(hr_image_names):
        name_root = hr_name.split(hr_suffix_name)[0]
        sr_name = osp.join(input_folder, name_root + sr_suffix_name)

        if not osp.exists(sr_name):
            print('ERROR: the corresponding SR image was not found: ', sr_name)
            continue

        hr_name = osp.join(input_folder, hr_name)

        hr_image = Image.open(hr_name)
        sr_image = Image.open(sr_name)

        assert (hr_image.size == sr_image.size)

        psnr = PSNR(sr_image, hr_image, shave_border)
        # print(i, ')PSNR for', name_root, '=', psnr)
        PSNRs.append(psnr)

    print('Average PSNR=', np.mean(PSNRs))

if __name__ == "__main__":
    main()
