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
def tf_distort_color(image):
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


class Dataset:

    def __init__(self, images_paths, labels, is_real, input_size, batch_size, params,
                 return_original=False):
        self.images_paths = images_paths
        self.input_size = input_size
        self.batch_size = batch_size
        self.params = params
        self.return_original = return_original

        self.loaded_images = []
        self.labels = Dataset.reassign_labels(labels)
        self.is_real = is_real

        if self.params['preload']:
            self.preload()
            if self.params['pretile']:
                self.pretile()

        self.images_indexes_per_class = collections.defaultdict(list)
        for index, label in enumerate(self.labels):
            self.images_indexes_per_class[label].append(index)

        if self.params['weighted_sampling']:
            self.calc_sampling_probs()

    def calc_sampling_probs(self):
        ''' Counts number of images per class and returns probability distribution so that
            distribution of images classes becomes uniform.
        '''

        frequency = {l: self.labels.count(l) for l in set(self.labels)}

        probs = np.empty((len(self.labels)), dtype=np.float32)
        for idx, l in enumerate(self.labels):
            probs[idx] = 1.0 / frequency[l]
        self.probs = probs / np.sum(probs)

    def preload(self):
        ''' Pre-loads images in RAM. '''

        for image_path in self.images_paths:
            self.loaded_images.append(cv2.imread(image_path))

    def pretile(self):
        ''' Pre-tiles images in RAM. Makes training faster but requires huge amount of RAM. '''

        tiled_labels = []
        tiled_is_real = []
        tiled_loaded_images = []

        for read_image, label, real in zip(self.loaded_images, self.labels, self.is_real):
            if not real:
                for n in range(2, self.params['max_tiling'] + 1):
                    image = self.tile(read_image, n)

                    tiled_labels.append(label)
                    tiled_is_real.append(real)
                    tiled_loaded_images.append(image)

        self.labels.extend(tiled_labels)
        self.is_real.extend(tiled_is_real)
        self.loaded_images.extend(tiled_loaded_images)


    def tile(self, image, n):
        ''' Tiles images taking their aspect ratios into account. '''

        aspect_ratio = image.shape[1] / image.shape[0]
        if aspect_ratio < 1:
            w_repeats = n
            h_repeats = max(1 if n != self.params['max_tiling'] else 2, int(n * aspect_ratio))
        else:
            h_repeats = n
            w_repeats = max(1 if n != self.params['max_tiling'] else 2, int(n / aspect_ratio))

        image = np.tile(image, (h_repeats, w_repeats, 1))

        fit_size = self.input_size * 3
        if image.shape[0] > fit_size or image.shape[1] > fit_size:
            image = fit_to_max_size(image, self.input_size * 3)

        return image

    def sample_index(self):
        ''' Samples indexes. '''

        choices = list(range(len(self.labels)))
        if self.params['weighted_sampling']:
            choices = np.random.choice(choices, len(self.labels), p=self.probs)
        elif self.params['shuffle']:
            np.random.shuffle(choices)

        # duplication is required for triplet loss at least.
        duplicated_choices = []
        for choice in choices:
            for _ in range(self.params['duplicate_n_times']):
                duplicated_choices.append(int(
                    np.random.choice(
                        self.images_indexes_per_class[self.labels[choice]],
                        1)))

        for choice in duplicated_choices:
            yield [choice]

    def read(self, index):
        ''' Reads an image from RAM or disk and returns it with corresponding class label. '''

        if self.params['preload']:
            image = self.loaded_images[index[0]].astype(np.float32)
        else:
            image = cv2.imread(self.images_paths[index[0]]).astype(np.float32)

        if not self.params['pretile'] and not self.is_real[index[0]]:
            n = random.randint(1, self.params['max_tiling'])
            image = self.tile(image, n)

        return image, self.labels[index[0]]

    def cv2_rotate(self, image):
        ''' Rotates images on random angle using opencv. '''

        c_xy = image.shape[1] / 2, image.shape[0] / 2
        angle = random.uniform(-self.params['add_rot_angle'],
                               self.params['add_rot_angle']) * 57.2958

        if self.params['rot90']:
            angle += random.randint(0, 3) * 180

        rotation_matrix = cv2.getRotationMatrix2D(c_xy, angle, 1)
        img_rotation = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return img_rotation

    def cv2_noise_and_blur(self, image):
        ''' Adds noise making image darker and blur.'''

        image = image.astype(np.float32)

        if self.params['apply_gray_noise'] and np.random.choice([True, False]):
            image = gray_noise(image)

        if self.params['blur'] and np.random.choice([True, False]):
            image = blur(image)

        return image

    def train_preprocess(self, choice):
        ''' Applies training preprocessing. '''

        original, label = tf.numpy_function(self.read, [choice], [tf.float32, tf.int64])
        image = tf_random_crop_and_resize(original, self.input_size)
        image, = tf.numpy_function(self.cv2_noise_and_blur, [image], [tf.float32])
        if self.params['horizontal_flip']:
            image = tf.image.random_flip_left_right(image)
        if self.params['vertical_flip']:
            image = tf.image.random_flip_up_down(image)
        if self.params['add_rot_angle'] > 0 or self.params['rot90']:
            image, = tf.numpy_function(self.cv2_rotate, [image], [tf.float32])
        image = tf_distort_color(image)
        image = preproces_image(image)
        if self.return_original:
            return image, label, original
        return image, label

    def __call__(self, *args, **kwargs):
        ''' Returns tf.data.Dataset instance as well as number of classes in training set. '''

        dataset = tf.data.Dataset.from_generator(self.sample_index, (tf.int32),
                                                 (tf.TensorShape([1])))
        dataset = dataset.map(self.train_preprocess,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if not self.return_original:
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            dataset = dataset.repeat()

        return dataset, len(set(self.labels))

    @staticmethod
    def create_from_list(path, input_size, batch_size, params, return_original=False):
        ''' Creates Dataset instance from path to images list.
            Images list has following format:
            <relative_path_to_image> <class_label>
        '''

        impaths, labels, is_real, _ = from_list(path)

        return Dataset(impaths, labels, is_real, input_size, batch_size, params, return_original)()

    @staticmethod
    def reassign_labels(labels):
        ''' Re-assign class labels so that they starts from 0 and ends with (num_classes - 1). '''

        unique_labels = list(set(labels))
        return [unique_labels.index(l) for l in labels]


def main():
    import argparse
    import time

    args = argparse.ArgumentParser()
    args.add_argument('--gallery', required=True)
    args.add_argument('--input_size', default=224, type=int)
    args.add_argument('--augmentation_config', required=True)
    args = args.parse_args()

    with open(args.augmentation_config) as f:
        augmentation_config = json.load(f)

    dataset, _ = Dataset.create_from_list(args.gallery, args.input_size, 1,
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
