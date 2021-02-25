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

""" This script allows you to train Text Recognition model. """

import argparse
import os
import time
import numpy as np
import tensorflow as tf

from text_recognition.model import TextRecognition
from text_recognition.dataset import Dataset


def parse_args():
    """ Parses input arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, required=True, help='Training annotation path.')
    parser.add_argument('--annotation_path_test', type=str, required=False, default='', help='Test annotation path.')
    parser.add_argument('--weights_path', type=str, help='Pretrained model weights.')
    parser.add_argument('--reg', action='store_true', help='Use weights regularization.')
    parser.add_argument('--backbone_dropout', type=float, default=0.0, help='Use dropout')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--num_steps', type=int, default=1000000)

    return parser.parse_args()

# pylint: disable=too-many-locals,too-many-statements
def main():
    """ Main training function. """

    args = parse_args()

    seq_length = 30
    batch_size = 64

    image_width, image_height = 120, 32

    handle = tf.placeholder(tf.string, shape=[])

    dataset_train = Dataset(args.annotation_path, image_width, image_height, batch_size=batch_size,
                            shuffle=True)

    iterator_train = dataset_train().make_initializable_iterator()

    if args.annotation_path_test != '':
        dataset_test = Dataset(args.annotation_path_test, image_width, image_height,
                               batch_size=batch_size, shuffle=False, repeat=1)
        iterator_test = dataset_test().make_initializable_iterator()

    iterator = tf.data.Iterator.from_string_handle(
        handle, dataset_train().output_types, dataset_train().output_shapes,
        dataset_train().output_classes)
    next_sample = iterator.get_next()

    is_training_ph = tf.placeholder(tf.bool)

    model = TextRecognition(is_training=is_training_ph, num_classes=dataset_train.num_classes,
                            backbone_dropout=args.backbone_dropout)
    model_out = model(inputdata=next_sample[0])

    ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=next_sample[1], inputs=model_out,
                                             sequence_length=seq_length * np.ones(batch_size)))

    reg_loss = tf.losses.get_regularization_loss()
    loss = ctc_loss

    if args.reg:
        loss += reg_loss

    decoded, _ = tf.nn.ctc_beam_search_decoder(model_out, seq_length * np.ones(batch_size),
                                               merge_repeated=False)

    edit_dist = tf.edit_distance(tf.cast(decoded[0], tf.int32), next_sample[1])
    crw = tf.nn.zero_fraction(edit_dist)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    starter_learning_rate = args.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000000, 0.1, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss=loss,
                                                                       global_step=global_step)

    # Set tf summary
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    model_descr = str(train_start_time)

    tboard_save_path = 'tboard/' + model_descr

    if not os.path.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    tf.summary.scalar(name='ctc_loss', tensor=ctc_loss)
    tf.summary.scalar(name='reg_loss', tensor=reg_loss)
    tf.summary.scalar(name='total_loss', tensor=loss)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    merge_summary_op = tf.summary.merge_all()

    test_acc_ph = tf.placeholder(dtype=np.float32)
    test_acc_summary = tf.summary.scalar(name='test_acc_ph', tensor=test_acc_ph)

    # Set saver configuration
    saver = tf.train.Saver(max_to_keep=1000)
    model_save_dir = 'model/' + model_descr
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_name = 'model_' + model_descr + '.ckpt'
    model_save_path = os.path.join(model_save_dir, model_name)

    summary_writer = tf.summary.FileWriter(tboard_save_path)

    with tf.Session() as sess:
        sess.run(iterator_train.initializer)
        if args.weights_path is None:
            print('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            print('Restore model from {:s}'.format(args.weights_path))
            saver.restore(sess=sess, save_path=args.weights_path)

        training_handle = sess.run(iterator_train.string_handle())
        if args.annotation_path_test != '':
            test_handle = sess.run(iterator_test.string_handle())

        for _ in range(args.num_steps):
            _, c, step, summary = sess.run([optimizer, ctc_loss, global_step, merge_summary_op],
                                           feed_dict={is_training_ph: True,
                                                      handle: training_handle})

            if step % 100 == 0:
                summary_writer.add_summary(summary=summary, global_step=step)
                print('Iter: {:d} cost= {:9f}'.format(step, c))

            if step % 1000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=global_step)

                if args.annotation_path_test:
                    sess.run(iterator_test.initializer)

                    correct = 0.0
                    for _ in range(len(dataset_test) // batch_size):
                        correct += sess.run(crw,
                                            feed_dict={is_training_ph: False, handle: test_handle})

                    test_accuracy = correct / (len(dataset_test) // batch_size)

                    print('Iter: {:d} cost= {:9f} TEST accuracy= {:9f}'.format(step, c,
                                                                               test_accuracy))

                    summary = sess.run(test_acc_summary, feed_dict={test_acc_ph: test_accuracy})
                    summary_writer.add_summary(summary=summary, global_step=step)


if __name__ == '__main__':
    main()
