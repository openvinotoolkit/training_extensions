""" This module performs training of text detection neural network. """

import argparse
import os
import test
import yaml

import numpy as np

import tensorflow as tf

from loss import ClassificationLoss, LinkageLoss
from model import pixel_link_model
from dataset import TFRecordDataset


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
    args.add_argument('--freeze_backbone', action='store_true')
    args.add_argument('--batch_size', type=int, default=24, help='Training batch size.')
    args.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    args.add_argument('--momentum', type=float, default=0.9,
                      help='The momentum for the MomentumOptimizer')
    args.add_argument('--weights_decay', type=float, default=0.0001, help='Weights decay.')
    args.add_argument('--weights', help='Path to pretrained weights.')
    args.add_argument('--train_dataset', required=True, help='Training dataset path.')
    args.add_argument('--test_dataset', help='Test dataset path.')
    args.add_argument('--train_resolution', type=int, nargs=2, default=[512, 512],
                      help='Training image resolution.')
    args.add_argument('--test_resolution', type=int, nargs=2, default=[1280, 768],
                      help='Test image resolution.')
    args.add_argument('--epochs_per_evaluation', type=int, default=1)

    return args


def config_initialization(args):
    """ Initializes training configuration. """

    strides = 4
    config = {
        'model_type': args.model_type,
        'freeze_backbone': args.freeze_backbone,
        'weights_decay': args.weights_decay,
        'batch_size': args.batch_size,
        'train_image_shape': args.train_resolution[::-1],
        'score_map_shape': [args.train_resolution[1] // strides,
                            args.train_resolution[0] // strides],
        'rotate': True,
        'rotation_prob': 0.5,
        'distort_color': True,
        'random_crop': True,
        'imagenet_preprocessing': args.model_type != 'mobilenet_v2_ext',
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
    }

    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    with open(os.path.join(args.train_dir, 'configuration.yaml'), 'w') as read_file:
        yaml.dump(config, read_file)

    return config


class ExponentialMovingAverageCallback(tf.keras.callbacks.Callback):
    """ Callback for Exponential Moving Average computation. """

    def __init__(self, model):
        super(ExponentialMovingAverageCallback, self).__init__()

        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.model = model
        self.ema.apply(self.model.trainable_variables)

    def on_batch_end(self, batch, logs=None):
        """ Exponential Moving Average computation callback. """

        self.ema.apply(self.model.trainable_variables)


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

    dataset, size = TFRecordDataset(args.train_dataset, config)()
    model = pixel_link_model(
        inputs=tf.keras.Input(shape=(args.train_resolution[1], args.train_resolution[0], 3)),
        config=config)
    ema_cb = ExponentialMovingAverageCallback(model)
    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=args.learning_rate,
                                                     momentum=args.momentum)
    model.compile(loss=[ClassificationLoss(config), LinkageLoss(config)], optimizer=optimizer)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    tf.keras.utils.plot_model(model, os.path.join(args.train_dir, 'model.png'))

    if args.test_dataset:
        model_test = pixel_link_model(tf.keras.Input(shape=args.test_resolution[::-1] + [3]),
                                      config=config)
        test_args = Args()
        test_args.resolution = tuple(args.test_resolution)
        dataset_test, _ = TFRecordDataset(args.test_dataset, config, test=True)()

    if args.weights:
        model.load_weights(args.weights)

    latest_checkpoint = tf.train.latest_checkpoint(args.train_dir)
    if latest_checkpoint is not None:
        checkpoint.restore(latest_checkpoint)
        # Here is a workaround how to save/load EMA weights.
        model.load_weights(get_weights_path(latest_checkpoint, is_ema=True))
        for var in model.trainable_variables:
            ema_cb.ema._averages[var].assign(var)

    with tf.summary.create_file_writer(args.train_dir + "/logs").as_default():
        while True:
            latest_checkpoint_before_fit = tf.train.latest_checkpoint(args.train_dir)
            if latest_checkpoint_before_fit is not None:
                model.load_weights(get_weights_path(latest_checkpoint_before_fit, is_ema=False))

            history = model.fit(dataset, epochs=args.epochs_per_evaluation,
                                steps_per_epoch=size // args.batch_size, callbacks=[ema_cb])
            checkpoint.save(os.path.join(args.train_dir, 'model'))

            latest_checkpoint_after_fit = tf.train.latest_checkpoint(args.train_dir)

            # Save weights.
            model.save_weights(get_weights_path(latest_checkpoint_after_fit, is_ema=False))

            # Save ema weights.
            for var in model.trainable_variables:
                var.assign(ema_cb.ema.average(var))
            model.save_weights(get_weights_path(latest_checkpoint_after_fit, is_ema=True))

            epoch = int(latest_checkpoint_after_fit.split('-')[-1])

            tf.summary.scalar('training/loss', data=history.history['loss'][-1], step=epoch)
            tf.summary.scalar('training/segm_logits_loss',
                              data=history.history['segm_logits_loss'][-1], step=epoch)
            tf.summary.scalar('training/link_logits_loss',
                              data=history.history['link_logits_loss'][-1], step=epoch)
            tf.summary.scalar('training/learning_rate', data=args.learning_rate, step=epoch)
            tf.summary.scalar('training/batch_size', data=args.batch_size, step=epoch)

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
