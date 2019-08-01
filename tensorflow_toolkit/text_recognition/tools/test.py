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

""" This script allows you to test Text Recognition model. """

import argparse

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import cv2

from text_recognition.model import TextRecognition
from text_recognition.dataset import Dataset


def parse_args():
    """ Parases input arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', required=True, help='Annotation path.')
    parser.add_argument('--weights_path', required=True, help='Model weights path.')
    parser.add_argument('--show', action='store_true', help='Show images.')

    return parser.parse_args()


def main():
    """ Main testing funciton. """

    args = parse_args()

    sequence_length = 30
    image_width = 120
    image_height = 32

    dataset = Dataset(args.annotation_path, image_width, image_height, repeat=1)
    next_sample = dataset().make_one_shot_iterator().get_next()

    model = TextRecognition(is_training=False, num_classes=dataset.num_classes)
    images_ph = tf.placeholder(tf.float32, [1, image_height, image_width, 1])
    model_out = model(inputdata=images_ph)
    decoded, _ = tf.nn.ctc_beam_search_decoder(model_out, sequence_length * np.ones(1),
                                               merge_repeated=False)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=args.weights_path)

        correct = 0.0
        dataset_len = len(dataset)
        for _ in tqdm(range(dataset_len)):
            images_batch, labels_batch = sess.run(next_sample)

            preds, _ = sess.run([decoded, model_out], feed_dict={images_ph: images_batch})

            try:
                predicted = Dataset.sparse_tensor_to_str(preds[0], dataset.int_to_char)[0]
                expected = Dataset.sparse_tensor_to_str(labels_batch, dataset.int_to_char)[
                    0].lower()
            except:
                print('Could not find a word')
                continue

            correct += 1 if predicted == expected else 0

            if args.show and predicted != expected:
                image = np.reshape(images_batch, [image_height, image_width, -1]).astype(np.uint8)
                cv2.imshow('image', image)
                print('pr, gt', predicted, expected)
                k = cv2.waitKey(0)
                if k == 27:
                    sess.close()
                    return

        print('accuracy', correct / dataset_len)

    return


if __name__ == '__main__':
    main()
