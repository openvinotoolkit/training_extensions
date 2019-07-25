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

import json
import math
import random

import cv2
import numpy as np
import tensorflow as tf

from textile.common import (max_central_square_crop, preproces_image, depreprocess_image, fit_to_max_size, from_list)


def blur(image):
    kernel = np.ones((3, 3), np.float32) / 9
    image = cv2.filter2D(image, -1, kernel)
    return image


def gray_noise(image):
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


#pylint: disable=R0915
def create_dataset(impaths, labels, is_real, input_size, batch_size, params, return_original=False):

    if params['weighted_sampling']:
        frequency = {}
        for l in labels:
            if l not in frequency:
                frequency[l] = 0
            frequency[l] += 1

        probs = list(range(len(impaths)))
        for idx, l in enumerate(labels):
            probs[idx] = 1.0 / frequency[l]

        probs = np.array(probs)
        probs = probs / np.sum(probs)

    assert math.log(params['duplicate_n_times'], 2) == int(math.log(params['duplicate_n_times'], 2))

    def random_number():
        choices = list(range(len(impaths)))
        if params['weighted_sampling']:
            choices = np.random.choice(choices, len(impaths), p=probs)
        elif params['shuffle']:
            np.random.shuffle(choices)

        for _ in range(int(math.log(params['duplicate_n_times'], 2))):
            choices = [x for t in zip(choices, choices) for x in t]

        for choise in choices:
            yield [choise]

    def cv2_preprocess(image):
        image = image.astype(np.float32)

        if params['apply_gray_noise'] and np.random.choice([True, False]):
            image = gray_noise(image)

        if params['fit_to_max_size']:
            image = fit_to_max_size(image, params['fit_to_max_size'])

        if params['blur'] and np.random.choice([True, False]):
            image = blur(image)

        return image

    def read(choice):
        original = cv2.imread(impaths[choice[0]])
        image = cv2_preprocess(original)

        if params['sample_original_prob'] > 0 or return_original:
            original = max_central_square_crop(original)
            original = cv2.resize(original, (input_size, input_size))
            original = preproces_image(original)
            original = original.astype(np.float32)
        else:
            original = np.zeros((), dtype=np.float32)

        return original, image, labels[choice[0]], is_real[choice[0]]

    def read_image(choice):
        original, image, label, is_real = tf.numpy_function(read, [choice], [tf.float32, tf.float32, tf.int64, tf.bool])
        return original, image, label, is_real

    def tf_horizontal_flip(original, image, label):
        image = tf.image.random_flip_left_right(image)
        return original, image, label

    def tf_vertical_flip(original, image, label):
        image = tf.image.random_flip_up_down(image)
        return original, image, label

    @tf.function
    def tf_tile(original, image, label, is_real):
        if tf.not_equal(is_real, tf.constant(True)):
            vtimes = tf.random.uniform((), 1, params['max_tiling'], dtype=tf.int32)
            htimes = tf.maximum(1, vtimes + tf.random.uniform((), -1, 1, dtype=tf.int32))
            image = tf.tile(image, [vtimes, htimes, 1])
        return original, image, label

    def cv2_rotate(image):
        c_xy = image.shape[1] / 2, image.shape[0] / 2
        angle = random.uniform(-params['add_rot_angle'], params['add_rot_angle']) * 57.2958

        if params['rot90']:
            angle += random.randint(0, 3) * 180

        rotation_matrix = cv2.getRotationMatrix2D(c_xy, angle, 1)
        img_rotation = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return img_rotation

    def random_rotate(original, image, label):
        if params['add_rot_angle'] > 0 or params['rot90']:
            image, = tf.numpy_function(cv2_rotate, [image], [tf.float32])
        return original, image, label

    def random_crop_and_resize(original, image, label):
        image = tf_random_crop_and_resize(image, input_size)
        return original, image, label

    def random_distort_color(original, image, label):
        image = distort_color(image)

        return original, image, label

    @tf.function
    def normalize(original, image, label):
        image = preproces_image(image)

        if params['sample_original_prob'] > 0:
            if tf.less(tf.random.uniform((), 0.0, 1.0), params['sample_original_prob']):
                return original, original, label

        return original, image, label

    def last(original, image, label):
        if return_original:
            return original, image, label
        return image, label

    dataset = tf.data.Dataset.from_generator(random_number, (tf.int32), (tf.TensorShape([1])))
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if params['max_tiling'] > 1:
        dataset = dataset.map(tf_tile, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(random_crop_and_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if params['vertical_flip']:
        dataset = dataset.map(tf_vertical_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if params['horizontal_flip']:
        dataset = dataset.map(tf_horizontal_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(random_rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(random_distort_color, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(last, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if return_original:
        pass
    else:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if params['repeat']:
        dataset = dataset.repeat()

    return dataset, len(set(labels))

def create_dataset_path(path, input_size, batch_size, params, return_original=False):
    impaths, labels, is_real, _ = from_list(path)

    return create_dataset(impaths, labels, is_real, input_size, batch_size, params, return_original)

def main():
    import argparse
    import time

    args = argparse.ArgumentParser()
    args.add_argument('--gallery_folder', required=True)
    args.add_argument('--input_size', default=128, type=int)
    args.add_argument('--augmentation_config', required=True)
    args = args.parse_args()

    with open(args.augmentation_config) as f:
        augmentation_config = json.load(f)

    dataset, _ = create_dataset_path(args.gallery_folder, args.input_size, 1, augmentation_config,
                                     return_original=True)

    t = time.time()
    for original, preprocessed, label in dataset.take(1000):
        assert original.shape
        cv2.imshow('original', depreprocess_image(original.numpy()))
        cv2.imshow('preprocessed', depreprocess_image(preprocessed.numpy()))
        print(label)
        if cv2.waitKey(0) == 27:
            break
    print(time.time() - t)


if __name__ == '__main__':
    main()
