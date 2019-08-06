"""
 Copyright (c) 2019 Intel Corporation

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

import collections
import json
import math
import random

import cv2
import numpy as np
import tensorflow as tf

from image_retrieval.common import preproces_image, depreprocess_image, fit_to_max_size, from_list


def blur(image):
    kernel = np.ones((3, 3), np.float32) / 9
    image = cv2.filter2D(image, -1, kernel)
    return image


def gray_noise(image):
    if np.mean(image) > 100:
        gray = np.random.uniform(0.0, 100.0, image.shape[0:2])
        gray3 = np.array([gray, gray, gray])
        gray3 = np.transpose(gray3, (1, 2, 0))
        gray3 = cv2.blur(gray3, ksize=(7, 7))
        image -= gray3
        image = np.clip(image, 0.0, 255.0)

    return image


@tf.function
def tf_random_crop_and_resize(image, input_size):
    min_size = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    crop_size = tf.random.uniform((), min_size // 2, min_size, dtype=tf.int32)

    crop = tf.image.random_crop(image, (crop_size, crop_size, 3))

    var_thr = 100

    for _ in tf.range(10):
        moments = tf.nn.moments(tf.reshape(crop, (-1, 3)), axes=0)

        if tf.less(tf.reduce_sum(moments[1]), tf.constant(var_thr, dtype=tf.float32)):
            crop = tf.image.random_crop(image, (crop_size, crop_size, 3))
        else:
            break

    moments = tf.nn.moments(tf.reshape(crop, (-1, 3)), axes=0)
    if tf.less(tf.reduce_sum(moments[1]), tf.constant(var_thr, dtype=tf.float32)):
        crop = tf.image.random_crop(image, (tf.shape(image)[0], tf.shape(image)[1], 3))

    crop = tf.cast(tf.expand_dims(crop, axis=0), tf.float32)
    crop = tf.image.resize(crop, (input_size, input_size))
    crop = tf.squeeze(crop, axis=0)

    return crop


@tf.function
def distort_color(image):
    """ Distorts color. """

    image = image / 255.0
    image = image[:, :, ::-1]

    brightness_max_delta = 16. / 255.

    color_ordering = tf.random.uniform([], maxval=5, dtype=tf.int32)
    if tf.equal(color_ordering, 0):
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif tf.equal(color_ordering, 1):
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.1)
    elif tf.equal(color_ordering, 2):
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif tf.equal(color_ordering, 3):
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=brightness_max_delta)

    image = tf.clip_by_value(image, 0.0, 1.0)
    image = image * 255
    image = image[:, :, ::-1]

    return image


# pylint: disable=R0915
def create_dataset(impaths, labels, is_real, input_size, batch_size, params, return_original=False):
    tiled_images = []
    tiled_images_labels = []
    tiled_images_is_real = []

    tiled_images_indexes_per_class = collections.defaultdict(list)

    for impath, label, real in zip(impaths, labels, is_real):
        read_image = cv2.imread(impath)

        tiled_images_indexes_per_class[label].append(len(tiled_images))
        tiled_images_labels.append(label)
        tiled_images.append(read_image)
        tiled_images_is_real.append(real)

        if not real:
            for tile in range(2, params['max_tiling'] + 1):
                aspect_ratio = read_image.shape[1] / read_image.shape[0]
                if aspect_ratio < 1:
                    w_repeats = tile
                    h_repeats = max(1 if tile != params['max_tiling'] else 2,
                                    int(tile * aspect_ratio))
                else:
                    h_repeats = tile
                    w_repeats = max(1 if tile != params['max_tiling'] else 2,
                                    int(tile / aspect_ratio))

                image = np.tile(read_image, (h_repeats, w_repeats, 1))

                image = fit_to_max_size(image, input_size * 2)

                tiled_images_indexes_per_class[label].append(len(tiled_images))
                tiled_images_labels.append(label)
                tiled_images.append(image)
                tiled_images_is_real.append(real)

    if params['weighted_sampling']:
        frequency = {}
        for l in tiled_images_labels:
            if l not in frequency:
                frequency[l] = 0
            frequency[l] += 1

        probs = list(range(len(tiled_images)))
        for idx, l in enumerate(tiled_images_labels):
            probs[idx] = 1.0 / frequency[l]

        probs = np.array(probs)
        probs = probs / np.sum(probs)

    assert math.log(params['duplicate_n_times'], 2) == int(math.log(params['duplicate_n_times'], 2))

    def random_number():
        choices = list(range(len(tiled_images)))
        if params['weighted_sampling']:
            choices = np.random.choice(choices, len(tiled_images), p=probs)
        elif params['shuffle']:
            np.random.shuffle(choices)

        ducplicated_choices = []
        for choice in choices:
            for _ in range(params['duplicate_n_times']):
                ducplicated_choices.append(int(
                    np.random.choice(tiled_images_indexes_per_class[tiled_images_labels[choice]],
                                     1)))

        for choise in ducplicated_choices:
            yield [choise]

    def read(choice):
        image = tiled_images[choice[0]].astype(np.float32)
        return image, tiled_images_labels[choice[0]]

    def cv2_rotate(image):
        c_xy = image.shape[1] / 2, image.shape[0] / 2
        angle = random.uniform(-params['add_rot_angle'], params['add_rot_angle']) * 57.2958

        if params['rot90']:
            angle += random.randint(0, 3) * 180

        rotation_matrix = cv2.getRotationMatrix2D(c_xy, angle, 1)
        img_rotation = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return img_rotation

    def cv2_noise_and_blur(image):
        image = image.astype(np.float32)

        if params['apply_gray_noise'] and np.random.choice([True, False]):
            image = gray_noise(image)

        if params['blur'] and np.random.choice([True, False]):
            image = blur(image)

        return image

    def train_preprocess(choice):
        original, label = tf.numpy_function(read, [choice], [tf.float32, tf.int64])
        image = tf_random_crop_and_resize(original, input_size)
        image, = tf.numpy_function(cv2_noise_and_blur, [image], [tf.float32])
        if params['horizontal_flip']:
            image = tf.image.random_flip_left_right(image)
        if params['vertical_flip']:
            image = tf.image.random_flip_up_down(image)
        if params['add_rot_angle'] > 0 or params['rot90']:
            image, = tf.numpy_function(cv2_rotate, [image], [tf.float32])
        image = distort_color(image)
        image = preproces_image(image)
        if return_original:
            return image, label, original
        return image, label

    dataset = tf.data.Dataset.from_generator(random_number, (tf.int32), (tf.TensorShape([1])))
    dataset = dataset.map(train_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if not return_original:
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()

    return dataset, len(set(tiled_images_labels))


def create_dataset_from_list(path, input_size, batch_size, params, return_original=False):
    impaths, labels, is_real, _ = from_list(path)

    return create_dataset(impaths, labels, is_real, input_size, batch_size, params, return_original)


def main():
    import argparse
    import time

    args = argparse.ArgumentParser()
    args.add_argument('--gallery_folder', required=True)
    args.add_argument('--input_size', default=224, type=int)
    args.add_argument('--augmentation_config', required=True)
    args = args.parse_args()

    with open(args.augmentation_config) as f:
        augmentation_config = json.load(f)

    dataset, _ = create_dataset_from_list(args.gallery_folder, args.input_size, 1,
                                          augmentation_config, True)

    t = time.time()
    for preprocessed, label, original in dataset.take(1000):
        cv2.imshow('preprocessed', depreprocess_image(preprocessed.numpy()))
        cv2.imshow('original', original.numpy().astype(np.uint8))
        print(label)
        if cv2.waitKey(0) == 27:
            break
    print(time.time() - t)


if __name__ == '__main__':
    main()
