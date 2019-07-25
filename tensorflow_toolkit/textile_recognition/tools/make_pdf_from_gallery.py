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
import json
import argparse
import fpdf
import cv2

from textile.dataset import fit_to_max_size


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--gallery_folder', required=True)
    args.add_argument('--output', required=True)

    return args.parse_args()


def main():
    args = parse_args()
    page_w = 160
    page_h = 200
    pdf = fpdf.FPDF('P', 'mm', (page_w, page_h))
    pdf.set_font('Arial', 'B', 10)

    num_images_per_page = 20
    per_row = 4
    delta_h = page_h / per_row
    delta_w = page_w / per_row

    image_list = sorted(os.listdir(args.gallery_folder))

    for index, image_name in enumerate(image_list[:1000]):
        print(index)

        if index % num_images_per_page == 0:
            pdf.add_page()

        row = index % num_images_per_page // per_row
        col = index % num_images_per_page % per_row

        impath = os.path.join(args.gallery_folder, image_name)

        x = delta_w * col + 5
        y = 10 + delta_h * row
        pdf.text(x, y - 1, "type: {}".format(index))
        pdf.image(impath, x, y, 20, 20)


    with open(args.output + '.json', 'w') as f:
        json.dump(image_list, f)

    pdf.output(args.output, "F")


def main_images():
    import numpy as np

    args = parse_args()
    page_h = 1600
    page_w = 2000

    num_images_per_page = 20
    per_row = 5
    delta_h = page_h // (num_images_per_page // per_row)
    delta_w = page_w // per_row

    max_size = 250

    image_list = sorted(os.listdir(args.gallery_folder))

    output_image = None

    os.makedirs(args.output, exist_ok=True)

    for index, image_name in enumerate(image_list):
        print(index)

        if index % num_images_per_page == 0:
            if output_image is not None:
                cv2.imwrite(os.path.join(args.output, 'image{}.png'.format(index // num_images_per_page)), output_image)
                # cv2.imshow('image', output_image)
                # cv2.waitKey(0)

            output_image = np.ones([page_h, page_w, 3], dtype=np.uint8) * 255

        row = index % num_images_per_page // per_row
        col = index % num_images_per_page % per_row

        impath = os.path.join(args.gallery_folder, image_name)

        image = cv2.imread(impath)
        image = fit_to_max_size(image, max_size)

        pad_width = ((0, max_size-image.shape[0]), (0, max_size-image.shape[1]), (0, 0))

        image = np.pad(image, pad_width, mode='constant', constant_values=255)
        image[0, :] = 0
        image[max_size-1, :] = 0
        image[:, 0] = 0
        image[:, max_size - 1] = 0


        x = delta_w * col + 50
        y = 50 + delta_h * row

        cv2.putText(output_image, "class: {}".format(index), (x, y -5), 0, 1.0, (0, 0, 0), 2)

        output_image[y:y+max_size, x:x+max_size] = image

    cv2.imwrite(os.path.join(args.output, 'image{}.png'.format(index // num_images_per_page)),
                output_image)

    with open(args.output + '.json', 'w') as f:
        json.dump(image_list, f)



if __name__ == '__main__':
    main_images()
