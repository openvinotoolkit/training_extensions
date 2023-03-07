# ~~~ Program: Ultrasound Original Images and Masks Extraction and Dataset Creation ~~~ #
# ~~~ Author: Raj Krishan Ghosh ~~~ #

import cv2
import numpy as np
import os.path

f = 'breast_ultrasound_simulation/stage1/Dataset_BUSI_with_GT/normal/'

if not os.path.exists('breast_ultrasound_simulation/stage1/dataset_intermediate_original/'):
    os.makedirs('breast_ultrasound_simulation/stage1/dataset_intermediate_original/')

for file in os.listdir(f):
    if 'mask' not in file:
        f_img = f + file
        img = cv2.imread(f_img, 0)

        if not os.path.exists('breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_images/'):
            os.makedirs('breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_images/')

        filename = 'breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_images/' + file
        cv2.imwrite(filename, img)

        img_mask = np.copy(img) * 0

        for file_mask in os.listdir(f):
            if ('mask' in file_mask) and (file[:-4] in file_mask):
                img_mask = img_mask + cv2.imread(f + file_mask, 0)

        if not os.path.exists('breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_masks/'):
            os.makedirs('breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_masks/')

        filename_mask = 'breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_masks/' + file
        cv2.imwrite(filename_mask, img_mask)

f = 'breast_ultrasound_simulation/stage1/Dataset_BUSI_with_GT/benign/'

for file in os.listdir(f):
    if 'mask' not in file:
        f_img = f + file
        img = cv2.imread(f_img, 0)

        if not os.path.exists('breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_images/'):
            os.makedirs('breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_images/')

        filename = 'breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_images/' + file
        cv2.imwrite(filename, img)

        img_mask = np.copy(img) * 0

        for file_mask in os.listdir(f):
            if ('mask' in file_mask) and (file[:-4] in file_mask):
                img_mask = img_mask + cv2.imread(f + file_mask, 0)

        if not os.path.exists('breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_masks/'):
            os.makedirs('breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_masks/')

        filename_mask = 'breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_masks/' + file
        cv2.imwrite(filename_mask, img_mask)

f = 'breast_ultrasound_simulation/stage1/Dataset_BUSI_with_GT/malignant/'

for file in os.listdir(f):
    if 'mask' not in file:
        f_img = f + file
        img = cv2.imread(f_img, 0)

        if not os.path.exists('breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_images/'):
            os.makedirs('breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_images/')

        filename = 'breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_images/' + file
        cv2.imwrite(filename, img)

        img_mask = np.copy(img) * 0

        for file_mask in os.listdir(f):
            if ('mask' in file_mask) and (file[:-4] in file_mask):
                img_mask = img_mask + cv2.imread(f + file_mask, 0)

        if not os.path.exists('breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_masks/'):
            os.makedirs('breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_masks/')

        filename_mask = 'breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_masks/' + file
        cv2.imwrite(filename_mask, img_mask)
