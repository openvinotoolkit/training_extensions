# ~~~ Program: Ultrasound Pseudo B-Mode 2D Stage 0 to Dataset Creation with Augmentation ~~~ #
# ~~~ Author: Raj Krishan Ghosh ~~~ #

import cv2
import numpy as np
import os.path

f_images = './dataset_intermediate_original/original_images/'
f_stage0 = './dataset_intermediate_stage_0/stage_0_masks/'
f_masks = './dataset_intermediate_original/original_masks/'

f_images_output = 'breast_ultrasound_simulation/stage1/dataset_intermediate_bus/images/'
f_masks_output = 'breast_ultrasound_simulation/stage1/dataset_intermediate_bus/masks/'
f_stage0_output = 'breast_ultrasound_simulation/stage1/dataset_intermediate_bus/stage0/'

f_images_inference_output = 'breast_ultrasound_simulation/stage1/DATASET_BUS_INFERENCE/images/'
f_masks_inference_output = 'breast_ultrasound_simulation/stage1/DATASET_BUS_INFERENCE/masks/'
f_stage0_inference_output = 'breast_ultrasound_simulation/stage1/DATASET_BUS_INFERENCE/stage0/'

f_images_output_final = 'breast_ultrasound_simulation/stage1/DATASET_BUS/images/'
f_masks_output_final = 'breast_ultrasound_simulation/stage1/DATASET_BUS/masks/'
f_stage0_output_final = 'breast_ultrasound_simulation/stage1/DATASET_BUS/stage0/'

n_h_pos = 10

if not os.path.exists(f_images_output):
    os.makedirs(f_images_output)
if not os.path.exists(f_masks_output):
    os.makedirs(f_masks_output)
if not os.path.exists(f_stage0_output):
    os.makedirs(f_stage0_output)

if not os.path.exists(f_images_inference_output):
    os.makedirs(f_images_inference_output)
if not os.path.exists(f_masks_inference_output):
    os.makedirs(f_masks_inference_output)
if not os.path.exists(f_stage0_inference_output):
    os.makedirs(f_stage0_inference_output)

max_width = 0
max_height = 0
number_of_images = 0

for file in sorted(os.listdir(f_images)):
    image_original = cv2.imread(f_images + file, 0)

    if len(image_original) > max_height:
        max_height = len(image_original)
    if len(image_original[0]) > max_width:
        max_width = len(image_original[0])
    number_of_images += 1

print(max_width)
print(max_height)
print(number_of_images)

for file in sorted(os.listdir(f_images)):
    image_original = cv2.imread(f_images + file, 0)
    mask_original = cv2.imread(f_masks + file, 0)
    stage0_original = cv2.imread(f_stage0 + file, 0)

    image = np.zeros((max_height, max_width), dtype=np.uint8)
    mask = np.zeros((max_height, max_width), dtype=np.uint8)
    stage0 = np.zeros((max_height, max_width), dtype=np.uint8)
    starting_width = int((len(image[0]) - len(image_original[0])) / 2)

    image[0:len(image_original), starting_width:(starting_width + len(image_original[0]))] = image_original
    mask[0:len(image_original), starting_width:(starting_width + len(image_original[0]))] = mask_original
    stage0[0:len(image_original), starting_width:(starting_width + len(image_original[0]))] = stage0_original

    image_scaled = cv2.resize(image[:-3, :], (int(max_width / 2), int((max_height - 3) / 2)), cv2.INTER_NEAREST)
    mask_scaled = cv2.resize(mask[:-3, :], (int(max_width / 2), int((max_height - 3) / 2)), cv2.INTER_NEAREST)
    stage0_scaled = cv2.resize(stage0[:-3, :], (int(max_width / 2), int((max_height - 3) / 2)), cv2.INTER_NEAREST)

    cv2.imwrite(f_images_output + file[:-4] + "n.png", image_scaled)
    cv2.imwrite(f_images_output + file[:-4] + "f.png", np.flip(image_scaled, 1))
    cv2.imwrite(f_masks_output + file[:-4] + "n.png", mask_scaled)
    cv2.imwrite(f_masks_output + file[:-4] + "f.png", np.flip(mask_scaled, 1))
    cv2.imwrite(f_stage0_output + file[:-4] + "n.png", stage0_scaled)
    cv2.imwrite(f_stage0_output + file[:-4] + "f.png", np.flip(stage0_scaled, 1))

    if file[-6] == "1":
        image_scaled = cv2.resize(image_original,
                                  (int(len(image_original[0]) / 4), int(len(image_original) / 4)),
                                  cv2.INTER_NEAREST)
        mask_scaled = cv2.resize(mask_original,
                                 (int(len(image_original[0]) / 4), int(len(image_original) / 4)),
                                 cv2.INTER_NEAREST)
        stage0_scaled = cv2.resize(stage0_original,
                                   (int(len(image_original[0]) / 4), int(len(image_original) / 4)),
                                   cv2.INTER_NEAREST)

        cv2.imwrite(f_images_inference_output + file[:-4] + "n.png", image_scaled)
        cv2.imwrite(f_masks_inference_output + file[:-4] + "n.png", mask_scaled)
        cv2.imwrite(f_stage0_inference_output + file[:-4] + "n.png", stage0_scaled)

count = 0
for file in sorted(os.listdir(f_images_output)):
    mask_temp = cv2.imread(f_masks_output + file, 0)
    if count == 0:
        mask_total = np.zeros((number_of_images * 2, len(mask_temp), len(mask_temp[0])), dtype=np.uint8)
    mask_total[count] = np.copy(mask_temp)

    count += 1

left_crop = 0
right_crop = 0
bottom_crop = 0

addition = 0
while addition == 0:
    addition = np.sum(mask_total[:, :, left_crop])
    left_crop += 1

addition = 0
while addition == 0:
    addition = np.sum(mask_total[:, :, len(mask_total[0, 0]) - right_crop - 1])
    right_crop += 1

addition = 0
while addition == 0:
    addition = np.sum(mask_total[:, len(mask_total[0]) - bottom_crop - 1, :])
    bottom_crop += 1

left_crop -= 2
right_crop -= 2
bottom_crop -= 2

print(left_crop)
print(right_crop)
print(bottom_crop)

for file in sorted(os.listdir(f_images_output)):
    image = cv2.imread(f_images_output + file, 0)
    mask = cv2.imread(f_masks_output + file, 0)
    stage0 = cv2.imread(f_stage0_output + file, 0)

    image_cropped = image[:(len(image) - bottom_crop), left_crop:(len(image[0]) - right_crop)]
    mask_cropped = mask[:(len(mask) - bottom_crop), left_crop:(len(mask[0]) - right_crop)]
    stage0_cropped = stage0[:(len(stage0) - bottom_crop), left_crop:(len(stage0[0]) - right_crop)]

    cv2.imwrite(f_images_output + file, image_cropped)
    cv2.imwrite(f_masks_output + file, mask_cropped)
    cv2.imwrite(f_stage0_output + file, stage0_cropped)

if not os.path.exists(f_images_output_final):
    os.makedirs(f_images_output_final)
if not os.path.exists(f_masks_output_final):
    os.makedirs(f_masks_output_final)
if not os.path.exists(f_stage0_output_final):
    os.makedirs(f_stage0_output_final)

for file in sorted(os.listdir(f_images_output)):
    image = cv2.imread(f_images_output + file, 0)
    mask = cv2.imread(f_masks_output + file, 0)
    stage0 = cv2.imread(f_stage0_output + file, 0)
    img_height = len(image)
    temp_sum = 0
    left_start = 0
    right_start = 0
    while temp_sum == 0:
        temp_sum = np.sum(image[:, left_start])
        left_start += 1
    temp_sum = 0
    right_start = 0
    while temp_sum == 0:
        temp_sum = np.sum(image[:, (len(image[0]) - right_start - 1)])
        right_start += 1
    if left_start == (right_start + 1):
        right_start += 1
    if right_start == (left_start + 1):
        left_start += 1
    if left_start != right_start:
        print("Error! " + str(left_start) + "_" + str(right_start))

    while (len(image[0, left_start:])) < img_height:
        left_start -= 1

    left_start = 0

    left_end = len(image[0]) - img_height - 1 - left_start
    step = float(left_end - left_start) / 10.0

    for i in range(n_h_pos):
        image_cropped = image[:, (left_start + int(i * step)): (left_start + int(i * step) + img_height)]
        image_scaled = cv2.resize(image_cropped, (int(img_height / 2), int(img_height / 2)), cv2.INTER_NEAREST)
        mask_cropped = mask[:, (left_start + int(i * step)): (left_start + int(i * step) + img_height)]
        mask_scaled = cv2.resize(mask_cropped, (int(img_height / 2), int(img_height / 2)), cv2.INTER_NEAREST)
        stage0_cropped = stage0[:, (left_start + int(i * step)): (left_start + int(i * step) + img_height)]
        stage0_scaled = cv2.resize(stage0_cropped, (int(img_height / 2), int(img_height / 2)), cv2.INTER_NEAREST)

        cv2.imwrite(f_images_output_final + file[:-4] + "_" + f"{i:02}" + ".png", image_scaled)
        cv2.imwrite(f_masks_output_final + file[:-4] + "_" + f"{i:02}" + ".png", mask_scaled)
        cv2.imwrite(f_stage0_output_final + file[:-4] + "_" + f"{i:02}" + ".png", stage0_scaled)
