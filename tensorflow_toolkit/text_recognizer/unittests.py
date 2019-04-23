""" This module contains unit tests. """

import os
import numpy as np
import unittest

import tensorflow as tf

from dataset import Dataset
from model import TextRecognition


class TestCreateAnnotaion(unittest.TestCase):
    """ Tests set for annotation. """

    def setUp(self):
        """ setUp method for tests. """

        self.image_width = 120
        self.image_height = 64
        self.dataset = \
            Dataset('./data/annotation.txt', self.image_width, self.image_height, repeat=1)

    def test_num_frames(self):
        """ Test for checking number of frames in annotation. """

        self.assertEqual(len(self.dataset), 64)

    def test_images_shapes(self):
        """ Test for checking images shapes. """

        get_next = self.dataset().make_one_shot_iterator().get_next()

        with tf.Session() as sess:
            x, _ = sess.run(get_next)
            assert x.shape == (1, self.image_height, self.image_width, 1)


class TestTraining(unittest.TestCase):
    """ Tests set for training. """

    def test_training_loss(self):
        """ Test for checking that training loss decreases. """

        seq_length = 30
        batch_size = 64

        self.image_width = 120
        self.image_height = 32

        self.dataset = Dataset('./data/annotation.txt', self.image_width, self.image_height,
                               repeat=None, batch_size=batch_size)
        model = TextRecognition(is_training=True, num_classes=self.dataset.num_classes)

        next_sample = self.dataset().make_one_shot_iterator().get_next()
        model_out = model(inputdata=next_sample[0])

        ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=next_sample[1], inputs=model_out,
                                                 sequence_length=seq_length * np.ones(batch_size)))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdadeltaOptimizer(0.1).minimize(loss=ctc_loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            l0 = sess.run(ctc_loss)
            print('loss before', l0)

            for _ in range(10):
                sess.run(optimizer)

            l1 = sess.run(ctc_loss)
            print('loss after', l1)
            assert l1 < l0


if __name__ == '__main__':
    unittest.main()
