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

import argparse
import json
import os

import cv2
import numpy as np
import tensorflow as tf
from pygit2 import Repository
from sklearn.metrics.pairwise import cosine_distances

from image_retrieval.dataset import Dataset, depreprocess_image
from image_retrieval.losses import am_softmax_loss, triplet_loss, AMSoftmaxLogits
from image_retrieval.metrics import test_model
from image_retrieval.model import keras_applications_mobilenetv2, keras_applications_resnet50


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--gallery', required=True, help='Gallery images list.')
    args.add_argument('--test_gallery', help='Test gallery images list.')
    args.add_argument('--test_images', help='Test images list.')
    args.add_argument('--input_size', type=int, default=224, help='Input image size.')
    args.add_argument('--train_dir', required=True, help='Training folder.')
    args.add_argument('--model_weights', required=False, help='Path to model weights.')
    args.add_argument('--steps_per_epoch', default=10000, type=int)
    args.add_argument('--loss', required=True)
    args.add_argument('--model', choices=['resnet50', 'mobilenet_v2'], required=True)
    args.add_argument('--max_iters', default=400000, type=int)
    args.add_argument('--lr_init', type=float, default=0.001)
    args.add_argument('--lr_drop_value', type=float, default=0.1)
    args.add_argument('--lr_drop_step', type=int, default=100000)
    args.add_argument('--dump_hard_examples', action='store_true')
    args.add_argument('--batch_size', type=int, default=256)
    args.add_argument('--augmentation_config', required=True)

    return args.parse_args()


def latest_checkpoint_number(train_dir):
    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if latest_checkpoint:
        return int(latest_checkpoint.split('-')[-1])
    return -1


def collect_hard_images(images, labels, distances, indices, positive):
    hard_examples = []

    already_in = set()

    for pair in indices:
        if positive:
            if labels[pair[0]] != labels[pair[1]]:
                continue
        else:
            if labels[pair[0]] == labels[pair[1]]:
                continue

        if (pair[0], pair[1]) in already_in or (pair[1], pair[0]) in already_in:
            continue
        else:
            already_in.add((pair[0], pair[1]))

        concatenated = np.concatenate((images[pair[0]], images[pair[1]]), axis=1)

        header = np.zeros((50, concatenated.shape[1], 3))

        text = str(labels[pair[0]]) + '-' + str(labels[pair[1]]) + ': ' + str(
            distances[pair[0], pair[1]])

        cv2.putText(header, text, (0, 50), 1, 2.0, (255, 255, 255), 2)

        concatenated = np.concatenate((header, concatenated), axis=0)

        concatenated = concatenated / 255.0

        hard_examples.append(concatenated)

        if len(hard_examples) == 10:
            break

    hard_examples = np.array(hard_examples)

    return hard_examples


def greatest_loss(images, labels, embeddings):
    arr = cosine_distances(embeddings, embeddings)
    args_max = np.dstack(np.unravel_index(np.argsort(-arr.ravel()),
                                          (embeddings.shape[0], embeddings.shape[0])))[0]

    args_min = args_max[::-1]

    np_images = depreprocess_image(images.numpy())[:, :, :, ::-1]
    np_labels = labels.numpy()

    hard_positives = collect_hard_images(np_images, np_labels, arr, args_max, True)
    hard_negatives = collect_hard_images(np_images, np_labels, arr, args_min, False)

    return hard_positives, hard_negatives


def collect_hard_examples(model, dataset, dir):
    embeddings_folder = os.path.join(dir, 'embs')
    os.makedirs(embeddings_folder, exist_ok=True)
    for x, y in dataset.take(1):
        predicted_embeddings = model.predict(x)
        return greatest_loss(x, y, predicted_embeddings)


def save_args(args, path):
    with open(path, 'w') as f:
        json.dump(args.__dict__, f)


def save_git_info(path):
    repo = Repository(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
    info = {
        "branch": repo.head.shorthand,
        "commit_hex": repo.revparse_single('HEAD').hex,
        "commit_message": repo.revparse_single('HEAD').message,
    }

    with open(path, 'w') as f:
        json.dump(info, f)


# pylint: disable=R0912,R0915
def main():
    args = parse_args()
    if args.model == 'resnet50':
        model = keras_applications_resnet50(
            tf.keras.layers.Input(shape=(args.input_size, args.input_size, 3)))
    elif args.model == 'mobilenet_v2':
        model = keras_applications_mobilenetv2(
            tf.keras.layers.Input(shape=(args.input_size, args.input_size, 3)))
    else:
        raise Exception('unknown model')

    with open(args.augmentation_config) as f:
        augmentation_config = json.load(f)

    dataset, num_classes = Dataset.create_from_list(args.gallery, args.input_size, args.batch_size,
                                                    augmentation_config)

    if args.model_weights:
        model.load_weights(args.model_weights)
    elif os.path.exists(args.train_dir):
        latest_checkpoint = tf.train.latest_checkpoint(args.train_dir)
        if latest_checkpoint:
            model.load_weights(latest_checkpoint)
            print('loaded', latest_checkpoint)
    else:
        os.makedirs(args.train_dir)

    save_args(args, os.path.join(args.train_dir, 'args.json'))
    save_git_info(os.path.join(args.train_dir, 'git_info.json'))

    if args.loss.startswith('amsoftmax'):
        _, s, m = args.loss.split('_')
        s, m = float(s), float(m)
        print(s, m)
        loss_function = am_softmax_loss(num_classes, s, m)
        training_model = tf.keras.Sequential([
            model,
            AMSoftmaxLogits(num_classes)
        ])
    elif args.loss.startswith('triplet'):
        _, margin = args.loss.split('_')
        margin = float(margin)
        loss_function = triplet_loss(margin=margin)
        training_model = model
    else:
        raise Exception('unknown loss')

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr_init,
        decay_steps=args.lr_drop_step,
        decay_rate=args.lr_drop_value,
        staircase=True)

    training_model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(lr_schedule))
    if args.model_weights:
        training_model.optimizer.iterations.assign(0)

    with tf.summary.create_file_writer(args.train_dir + "/logs").as_default():
        while True:
            cur_step = training_model.optimizer.iterations.numpy()
            print('cur_step', cur_step)
            lr = training_model.optimizer.lr(cur_step).numpy()
            print('lr', lr)

            history = training_model.fit(dataset, steps_per_epoch=args.steps_per_epoch)

            cur_step = training_model.optimizer.iterations.numpy()
            lr = training_model.optimizer.lr(cur_step).numpy()

            # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            tf.summary.scalar('training/loss', data=history.history['loss'][-1], step=cur_step)
            tf.summary.scalar('training/lr', data=lr, step=cur_step)
            tf.summary.scalar('training/batch_size', data=args.batch_size, step=cur_step)

            save_to = os.path.join(args.train_dir, 'weights-{}'.format(cur_step))
            model.save_weights(save_to)

            print('Saved: {}'.format(save_to))

            if args.dump_hard_examples:
                hard_positives, hard_negatives = collect_hard_examples(model, dataset,
                                                                       args.train_dir)

                hard_positives = tf.convert_to_tensor(hard_positives)
                hard_negatives = tf.convert_to_tensor(hard_negatives)

                # pylint: disable=redundant-keyword-arg
                tf.summary.image('hard_positives', hard_positives, cur_step, max_outputs=10)
                tf.summary.image('hard_negatives', hard_negatives, cur_step, max_outputs=10)

            if args.test_images:
                if args.test_gallery:
                    gallery = args.test_gallery
                else:
                    gallery = args.gallery

                top1, top5, top10, mean_pos = test_model(model_path=None, model_backend=None,
                                                         model=model,
                                                         gallery_path=gallery,
                                                         test_images=args.test_images,
                                                         input_size=args.input_size)

                tf.summary.scalar('test/top1', data=top1, step=cur_step)
                tf.summary.scalar('test/top5', data=top5, step=cur_step)
                tf.summary.scalar('test/top10', data=top10, step=cur_step)
                tf.summary.scalar('test/mean_pos', data=mean_pos, step=cur_step)

            if cur_step > args.max_iters:
                break


if __name__ == '__main__':
    main()
