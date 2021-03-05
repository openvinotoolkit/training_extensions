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

""" This module performs training of text detection neural network. """

import argparse
import os
import test
import yaml

import tensorflow as tf
import numpy as np

from text_detection.loss import ClassificationLoss, LinkageLoss
from text_detection.model import pixel_link_model
from text_detection.dataset import TFRecordDataset


def arg_parser():
    """ Parses arguments. """

    args = argparse.ArgumentParser()
    args.add_argument('--train_dir', required=True, help='Training folder.')
    args.add_argument('--model_type', choices=['mobilenet_v2_ext',
                                               'ka_vgg16',
                                               'ka_resnet50',
                                               'ka_mobilenet_v2_1_0',
                                               'ka_mobilenet_v2_1_4',
                                               'ka_xception'],
                      required=True)
    args.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    args.add_argument('--momentum', type=float, default=0.9,
                      help='The momentum for the MomentumOptimizer')
    args.add_argument('--weights', help='Path to pretrained weights.')
    args.add_argument('--train_dataset', required=True, help='Training dataset path.')
    args.add_argument('--test_dataset', help='Test dataset path.')
    args.add_argument('--test_resolution', type=int, nargs=2, default=[1280, 768],
                      help='Test image resolution.')
    args.add_argument('--epochs_per_evaluation', type=int, default=1)
    args.add_argument('--num_epochs', type=int, default=1000000)
    args.add_argument('--config', required=True, help='Path to configuration file.')

    return args


def config_initialization(args):
    """ Initializes training configuration. """

    with open(args.config) as opened_file:
        config = yaml.load(opened_file)

    config['model_type'] = args.model_type
    config['imagenet_preprocessing'] = args.model_type != 'mobilenet_v2_ext'

    return config


def save_config(config, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, 'configuration.yaml'), 'w') as read_file:
        yaml.dump(config, read_file)


class ExponentialMovingAverageCallback(tf.keras.callbacks.Callback):
    """ Callback for Exponential Moving Average computation. """

    def __init__(self, model, epoch):
        super(ExponentialMovingAverageCallback, self).__init__()

        self.model = model
        self.decay = 0.9999
        self.epoch = epoch
        self.averages = {}


        self.num_updates = 0
        with tf.name_scope('ema'):
            for var in model.trainable_variables:
                self.averages[var] = tf.Variable(var)

    # pylint: disable=unused-argument
    def on_batch_end(self, batch, logs=None):
        """ Exponential Moving Average computation callback. """
        if self.epoch == 0:
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        else:
            decay = self.decay

        for var in self.model.trainable_variables:
            self.averages[var].assign(decay * self.averages[var] + (1 - decay) * var)

        self.num_updates += 1

    def copy_weights_from_model(self):
        for var in self.model.trainable_variables:
            self.averages[var].assign(var)

    def copy_weights_to_model(self):
        for var in self.model.trainable_variables:
            var.assign(self.averages[var])


class Args:
    """ Test arguments. """

    def __init__(self):
        self.weights = None
        self.resolution = None
        self.imshow_delay = -1


def train(args, config):
    """ This function performs training of text detection neural network. """

    def get_weights_path(latest_checkpoint, is_ema):
        base_name = os.path.basename(latest_checkpoint)
        path = os.path.join(args.train_dir, 'weights',
                            base_name + ('.ema' if is_ema else '') + '.save_weights')
        return path

    def get_epoch(latest_checkpoint):
        if latest_checkpoint is None:
            return 0
        return int(latest_checkpoint.split('-')[-1])

    save_config(config, args.train_dir)
    config['num_replicas'] = 1

    dataset, size = TFRecordDataset(args.train_dataset, config)()

    model = pixel_link_model(
        inputs=tf.keras.Input(shape=config['train_image_shape'] + [3]),
        config=config)

    loss = [ClassificationLoss(config), LinkageLoss(config)]

    optimizer = tf.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum)
    model.compile(loss=loss, optimizer=optimizer)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    if args.weights:
        model.load_weights(args.weights)

    latest_checkpoint = tf.train.latest_checkpoint(args.train_dir)
    if latest_checkpoint is not None:
        checkpoint.restore(latest_checkpoint)
        # Here is a workaround how to save/load EMA weights.
        model.load_weights(get_weights_path(latest_checkpoint, is_ema=True))

    ema_cb = ExponentialMovingAverageCallback(model, get_epoch(latest_checkpoint))
    ema_cb.copy_weights_from_model()

    if args.test_dataset:
        model_test = pixel_link_model(tf.keras.Input(shape=args.test_resolution[::-1] + [3]),
                                      config=config)
        test_args = Args()
        test_args.resolution = tuple(args.test_resolution)
        dataset_test, _ = TFRecordDataset(args.test_dataset, config, test=True)()

    with tf.summary.create_file_writer(args.train_dir + "/logs").as_default():
        for _ in range(int(np.ceil(args.num_epochs / args.epochs_per_evaluation))):
            latest_checkpoint_before_fit = tf.train.latest_checkpoint(args.train_dir)
            if latest_checkpoint_before_fit is not None:
                model.load_weights(get_weights_path(latest_checkpoint_before_fit, is_ema=False))

            history = model.fit(dataset, epochs=args.epochs_per_evaluation,
                                steps_per_epoch=size // config['batch_size'], callbacks=[ema_cb])
            checkpoint.save(os.path.join(args.train_dir, 'model'))

            latest_checkpoint_after_fit = tf.train.latest_checkpoint(args.train_dir)

            # Save weights.
            model.save_weights(get_weights_path(latest_checkpoint_after_fit, is_ema=False))

            # Save ema weights.
            ema_cb.copy_weights_to_model()
            model.save_weights(get_weights_path(latest_checkpoint_after_fit, is_ema=True))

            epoch = get_epoch(latest_checkpoint_after_fit)

            tf.summary.scalar('training/loss', data=history.history['loss'][-1], step=epoch)
            tf.summary.scalar('training/segm_logits_loss',
                              data=history.history['segm_logits_loss'][-1] * config['num_replicas'],
                              step=epoch)
            tf.summary.scalar('training/link_logits_loss',
                              data=history.history['link_logits_loss'][-1] * config['num_replicas'],
                              step=epoch)
            tf.summary.scalar('training/learning_rate', data=args.learning_rate, step=epoch)
            tf.summary.scalar('training/batch_size', data=config['batch_size'], step=epoch)

            if args.test_dataset:
                test_args.weights = get_weights_path(latest_checkpoint_after_fit, is_ema=False)
                recall, precision, hmean = test.test(test_args, config, model=model_test,
                                                     dataset=dataset_test)
                print('{} (recall, precision, hmean) = ({:.4f}, {:.4f}, {:.4f})'.format(
                    test_args.weights, recall, precision, hmean))
                tf.summary.scalar('common/hmean', data=hmean, step=epoch)
                tf.summary.scalar('common/precision', data=precision, step=epoch)
                tf.summary.scalar('common/recall', data=recall, step=epoch)

                test_args.weights = get_weights_path(latest_checkpoint_after_fit, is_ema=True)
                recall, precision, hmean = test.test(test_args, config, model=model_test,
                                                     dataset=dataset_test)
                print('{} (recall, precision, hmean) = ({:.4f}, {:.4f}, {:.4f})'.format(
                    test_args.weights, recall, precision, hmean))
                tf.summary.scalar('ema/hmean', data=hmean, step=epoch)
                tf.summary.scalar('ema/precision', data=precision, step=epoch)
                tf.summary.scalar('ema/recall', data=recall, step=epoch)


def main():
    """ Main function. """

    args = arg_parser().parse_args()
    config = config_initialization(args)
    train(args, config)


if __name__ == '__main__':
    main()
