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

""" This module contains Dataset class that operates with training data. """

import os
import numpy as np
import tensorflow as tf
import cv2


class Dataset():
    """ Class for working with training datasets. """

    def __init__(self, annotation_path, image_width, image_height, batch_size=1, shuffle=False, repeat=None):

        impaths, labels = Dataset.parse_datasets_arg(annotation_path)
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.size = len(impaths)
        self.char_to_int, self.int_to_char, self.num_classes = Dataset.create_character_maps()

        dataset = tf.data.Dataset.from_tensor_slices((impaths, labels))
        if shuffle:
            dataset = dataset.shuffle(len(impaths), reshuffle_each_iteration=True)
        dataset = dataset.map(
            lambda filename, label: tuple(
                tf.py_func(self.read_py_function, [filename, label], [tf.float32, tf.string])))
        dataset = dataset.map(
            lambda image, label: tuple(
                tf.py_func(self.convert_labels_to_int32_array, [image, label],
                           [tf.float32, tf.int32])))
        dataset = dataset.map(
            lambda image, label: tuple((image, self.to_sparse_tensor(label))))

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(self.set_shapes)
        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

        self.dataset = dataset

    def __call__(self):
        return self.dataset

    def __len__(self):
        return self.size

    def convert_labels_to_int32_array(self, image, label):
        """ Converts text to integer representation. """

        values = np.array([self.char_to_int[y] for y in label.decode('utf-8').lower()],
                          dtype=np.int32)
        return image, values

    def set_shapes(self, image, labels):
        """ Sets shapes for tensors. """

        image.set_shape([self.batch_size, self.image_height, self.image_width, 1])
        return image, labels

    def read_py_function(self, filename, label):
        """ Reads and pre-processes an image. """

        try:
            image = cv2.imread(filename.decode('utf-8'), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.astype(np.float32)
            image = cv2.resize(image, (self.image_width, self.image_height))
            image = np.expand_dims(image, axis=-1)
        except:
            print(filename)
            print(image.shape)
            raise Exception
        return image, label

    @staticmethod
    def create_character_maps():
        """ Creates character-to-int and int-to-character maps. """

        alfabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        char_to_int = {}
        int_to_char = []
        for i, l in enumerate(alfabet):
            char_to_int[l] = i
            int_to_char.append(l)
        return char_to_int, int_to_char, len(char_to_int) + 1

    @staticmethod
    def parse_datasets_arg(annotation_path):
        """ Parses datasets argument. """

        impaths = []
        labels = []
        for annpath in annotation_path.split(','):
            annotation_folder = os.path.dirname(annpath)
            with open(annpath, encoding="utf-8-sig") as f:
                content = np.array([line.strip().split() for line in f.readlines()])
                impaths_local = content[:, 0]
                impaths_local = [os.path.join(annotation_folder, line) for line in impaths_local]
                labels_local = content[:, 1]
                impaths.extend(impaths_local)
                labels.extend(labels_local)

        return impaths, labels

    @staticmethod
    def to_sparse_tensor(dense_tensor):
        """ Converts dense tensor to sparse. """

        indices = tf.where(tf.not_equal(dense_tensor, -1))
        values = tf.gather_nd(dense_tensor, indices)
        shape = tf.shape(dense_tensor, out_type=tf.int64)
        return tf.SparseTensor(indices, values, shape)

    @staticmethod
    def sparse_tensor_to_str(sparse_tensor, int_to_char):
        """ Converts sparse tensor to text string. """

        indices_set = set(sparse_tensor.indices[:, 0])
        result = {}
        for ind in indices_set:
            elements = sparse_tensor.indices[:, 0] == ind
            result[ind] = ''.join(
                [int_to_char[tmp] for tmp in sparse_tensor.values[np.ix_(elements)]])
        return result
