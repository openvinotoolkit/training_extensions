# ~~~ Program: Ultrasound Stage 0 Images and Masks Outputs and Dataset Creation ~~~ #
# ~~~ Author: Raj Krishan Ghosh ~~~ #

import cv2
from PolarPseudoBMode2D import generate_pseudo_b_mode_2d
import os.path

if not os.path.exists('breast_ultrasound_simulation/stage1/dataset_intermediate_stage_0/stage_0_masks/'):
    os.makedirs('breast_ultrasound_simulation/stage1/dataset_intermediate_stage_0/stage_0_masks/')

input_masks_path = 'breast_ultrasound_simulation/stage1/dataset_intermediate_original/original_masks/'
output_masks_path = 'breast_ultrasound_simulation/stage1/dataset_intermediate_stage_0/stage_0_masks/'

for file in sorted(os.listdir(input_masks_path)):
    f_mask = input_masks_path + file

    input_tensor = cv2.imread(f_mask, 0)
    output_tensor = generate_pseudo_b_mode_2d(input_tensor, low=0.4, mid=0.0, high=0.02,
                                              f0=20e6, sigma_x=5.0, sigma_y=2.0, speckle_variance=0.01, gamma=0.06)

    cv2.imwrite(output_masks_path + file, output_tensor)
    print("Done: " + file)
