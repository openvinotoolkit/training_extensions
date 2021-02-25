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

""" This module contains unit tests. """

import os
import unittest
import tempfile

import numpy as np

import tensorflow as tf

from text_detection.annotation import TextDetectionDataset, write_to_tfrecords
from text_detection.dataset import  TFRecordDataset
from text_detection.loss import ClassificationLoss, LinkageLoss
from text_detection.model import pixel_link_model
from text_detection.metrics import test


class TestCreateAnnotaion(unittest.TestCase):
    """ Tests set for annotation. """

    def setUp(self):
        """ setUp method for tests. """

        self.folder = './data'
        self.dataset = TextDetectionDataset.read_from_toy_dataset(self.folder)

    def test_num_frames(self):
        """ Test for checking number of frames in annotation. """

        self.assertEqual(len(self.dataset.annotation), 20)

    def test_image_paths(self):
        """ Test for checking correctness of images paths in annotation. """

        for index, frame in enumerate(self.dataset.annotation):
            self.assertEqual(frame['image_path'],
                             os.path.join(self.folder, 'img_{}.jpg'.format(index // 4 + 1)))

    def test_bboxes_nums(self):
        """ Test for checking number of bounding boxes in annotation. """

        bboxes_num = [8, 8, 8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4]
        for index, frame in enumerate(self.dataset.annotation):
            self.assertEqual(bboxes_num[index], len(frame['bboxes']))


class TestWriteAndReadAnnotaion(unittest.TestCase):
    """ Tests set for annotation io. """

    def setUp(self):
        """ setUp method for tests. """

        self.folder = './data'
        dataset = TextDetectionDataset.read_from_toy_dataset(self.folder)
        _, path = tempfile.mkstemp()
        path += '.json'
        dataset.write(path)
        self.dataset = TextDetectionDataset(path)

    def test_num_frames(self):
        """ Test for checking number of frames in annotation. """

        self.assertEqual(len(self.dataset.annotation), 20)

    def test_image_paths(self):
        """ Test for checking correctness of images paths in annotation. """

        for index, frame in enumerate(self.dataset.annotation):
            self.assertEqual(frame['image_path'],
                             os.path.join(self.folder, 'img_{}.jpg'.format(index // 4 + 1)))

    def test_bboxes_nums(self):
        """ Test for checking number of bounding boxes in annotation. """

        bboxes_num = [8, 8, 8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4]
        for index, frame in enumerate(self.dataset.annotation):
            self.assertEqual(bboxes_num[index], len(frame['bboxes']))


class TestCreateTFRecordDataset(unittest.TestCase):
    """ Tests set for TFRecordDataset. """

    def setUp(self):
        """ setUp method for tests. """

        self.folder = './data'
        dataset = TextDetectionDataset.read_from_toy_dataset(self.folder)
        _, path = tempfile.mkstemp()
        path += '.json'
        dataset.write(path)

        _, self.output_path = tempfile.mkstemp()
        self.output_path += '.tfrecord'
        write_to_tfrecords(output_path=self.output_path, datasets=[path])

    def test_dataset_size(self):
        """ Test for checking dataset size. """

        _, size = TFRecordDataset(self.output_path, {'model_type': 'mobilenet_v2_ext'}, test=True)()
        self.assertEqual(size, 20)

    def test_validation_dataset_image_shape(self):
        """ Test for checking validation dataset images shapes. """

        dataset, _ = TFRecordDataset(self.output_path, {'model_type': 'mobilenet_v2_ext'}, test=True)()
        for image, _, _ in dataset:
            self.assertEqual(image.numpy().shape, (768, 1280, 3))

    def test_training_dataset_image_shape(self):
        """ Test for checking training dataset images shapes. """

        config = {
            'model_type': 'mobilenet_v2_ext',
            'imagenet_preprocessing': False,
            'batch_size': 2,
            'weights_decay': 0.00001,
            'train_image_shape': [512, 512],
            'score_map_shape': [128, 128],
            'rotate': True,
            'rotation_prob': 0.5,
            'distort_color': True,
            'random_crop': True,
            'min_object_covered': 0.1,
            'bbox_crop_overlap': 0.2,
            'crop_aspect_ratio_range': (0.5, 2.),
            'area_range': [0.1, 1],
            'using_shorter_side_filtering': True,
            'min_shorter_side': 10,
            'max_shorter_side': np.infty,
            'min_area': 300,
            'min_height': 10,
            'max_neg_pos_ratio': 3,
            'num_neighbours': 8,
            'num_classes': 2,
            'ignore_label': -1,
            'background_label': 0,
            'text_label': 1,
            'num_replicas': 1
        }

        dataset, _ = TFRecordDataset(self.output_path, config, test=False)()
        counter = 0
        for image, _ in dataset:
            self.assertEqual(image.numpy().shape, (2, 512, 512, 3))
            counter += 1
            if counter > 20:
                break

        self.assertEqual(counter, 21)


class TestTraining(unittest.TestCase):
    """ Tests set for training. """

    def setUp(self):
        """ setUp method for tests. """

        self.folder = './data'
        dataset = TextDetectionDataset.read_from_toy_dataset(self.folder)
        _, path = tempfile.mkstemp()
        path += '.json'
        dataset.write(path)

        _, self.output_path = tempfile.mkstemp()
        self.output_path += '.tfrecord'
        write_to_tfrecords(output_path=self.output_path, datasets=[path])

        self.config = {
            'model_type': 'mobilenet_v2_ext',
            'imagenet_preprocessing': False,
            'batch_size': 5,
            'weights_decay': 0.00001,
            'train_image_shape': [256, 256],
            'score_map_shape': [64, 64],
            'rotate': True,
            'rotation_prob': 0.5,
            'distort_color': True,
            'random_crop': True,
            'min_object_covered': 0.1,
            'bbox_crop_overlap': 0.2,
            'crop_aspect_ratio_range': (0.5, 2.),
            'area_range': [0.1, 1],
            'using_shorter_side_filtering': True,
            'min_shorter_side': 10,
            'max_shorter_side': np.infty,
            'min_area': 300,
            'min_height': 10,
            'max_neg_pos_ratio': 3,
            'num_neighbours': 8,
            'num_classes': 2,
            'ignore_label': -1,
            'background_label': 0,
            'text_label': 1,
            'num_replicas': 1
        }

        self.model = pixel_link_model(tf.keras.Input(shape=(256, 256, 3)), self.config)
        optimizer = tf.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
        self.model.compile(loss=[ClassificationLoss(self.config),
                                 LinkageLoss(self.config)], optimizer=optimizer)

        _, self.saved_weights = tempfile.mkstemp()
        self.saved_weights = self.saved_weights + '-1'
        self.model.save_weights(self.saved_weights)
        self.history = None

    def test_loss_history(self):
        """ Test for checking loss history. """

        dataset, _ = TFRecordDataset(self.output_path, self.config, test=False)()
        self.history = self.model.fit(dataset, epochs=10, steps_per_epoch=4)
        self.assertGreater(self.history.history['loss'][0], self.history.history['loss'][-1])

    def test_eval_can_be_run(self):
        """ Test for checking an ability to run evaluation. """

        class Args:
            """ Arguments for evaluation. """

            def __init__(self, weights):
                self.imshow_delay = -1
                self.weights = weights
                self.resolution = (256, 256)

        dataset, _ = TFRecordDataset(self.output_path, self.config, test=True)()
        test(Args(self.saved_weights), self.config, model=self.model, dataset=dataset)


if __name__ == '__main__':
    unittest.main()
