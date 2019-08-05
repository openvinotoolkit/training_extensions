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

import os
import json
import argparse
from pygit2 import Repository
from sklearn.metrics.pairwise import cosine_distances
import tensorflow as tf
import numpy as np
import cv2

from textile.losses import am_softmax_loss, triplet_loss

from textile.model import keras_applications_mobilenetv2, keras_applications_resnet50
from textile.dataset import create_dataset_path, depreprocess_image

from textile.metrics import test_model


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--gallery_folder', required=True, help='Gallery images folder.')
    args.add_argument('--test_gallery_folder', help='Gallery images folder.')
    args.add_argument('--test_images_folder', help='Gallery images folder.')
    args.add_argument('--input_size', type=int, default=128, help='Input image size.')
    args.add_argument('--train_dir', required=True, help='Training folder.')
    args.add_argument('--model_weights', required=False, help='Path to model weights.')
    args.add_argument('--steps_per_epoch', default=10000, type=int)
    args.add_argument('--loss', required=True)
    args.add_argument('--model', choices=['resnet50', 'mobilenet_v2'], required=True)
    args.add_argument('--max_iters', default=400000, type=int)
    args.add_argument('--lr_init', type=float, default=0.001)
    args.add_argument('--lr_drop_value', type=float, default=0.1)
    args.add_argument('--lr_drop_step', type=int, default=100000)
    args.add_argument('--dump_embeddings', action='store_true')
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

        c = np.concatenate((images[pair[0]], images[pair[1]]), axis=1)

        header = np.zeros((50, c.shape[1], 3))

        text = str(labels[pair[0]]) + '-' + str(labels[pair[1]]) + ': ' + str(
            distances[pair[0], pair[1]])

        cv2.putText(header, text, (0, 50), 1, 2.0, (255, 255, 255), 2)

        c = np.concatenate((header, c), axis=0)

        c = c / 255.0

        hard_examples.append(c)

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


def dump_embeddings(model, dataset, dir, batches=10):
    embeddings_folder = os.path.join(dir, 'embs')
    os.makedirs(embeddings_folder, exist_ok=True)
    embeddings_path = os.path.join(embeddings_folder, 'embeddings.tsv')
    labels_path = os.path.join(embeddings_folder, 'labels.tsv')
    with open(embeddings_path, 'w') as f1, open(labels_path, 'w') as f2:
        for x, y in dataset.take(batches):
            predicted_embeddings = model.predict(x)

            return greatest_loss(x, y, predicted_embeddings)

            for i in range(predicted_embeddings.shape[0]):
                f1.write('\t'.join([str(x) for x in predicted_embeddings[i]]) + '\n')
                f2.write(str(y[i].numpy()) + '\n')


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
    if args.model == 'mobilenet_v2':
        model = keras_applications_mobilenetv2(
            tf.keras.layers.Input(shape=(args.input_size, args.input_size, 3)))

    with open(args.augmentation_config) as f:
        augmentation_config = json.load(f)

    dataset, num_classes = create_dataset_path(args.gallery_folder, args.input_size,
                                               args.batch_size,
                                               augmentation_config)

    if args.model_weights:
        model.load_weights(args.model_weights)

    if os.path.exists(args.train_dir):
        latest_checkpoint = tf.train.latest_checkpoint(args.train_dir)
        if latest_checkpoint:
            assert not args.model_weights
            model.load_weights(latest_checkpoint)
            print('loaded', latest_checkpoint)
    else:
        os.makedirs(args.train_dir)

    save_args(args, os.path.join(args.train_dir, 'args.json'))
    save_git_info(os.path.join(args.train_dir, 'git_info.json'))

    if args.loss.startswith('focal-amsoftmax'):
        _, s, m, alpha, gamma = args.loss.split('_')
        s, m, alpha, gamma = float(s), float(m), float(alpha), float(gamma)
        print(s, m, alpha, gamma)
        loss_function = am_softmax_loss(num_classes, s, m, alpha, gamma)
    elif args.loss.startswith('amsoftmax'):
        _, s, m = args.loss.split('_')
        s, m = float(s), float(m)
        print(s, m)
        loss_function = am_softmax_loss(num_classes, s, m)
    elif args.loss.startswith('triplet'):
        _, margin = args.loss.split('_')
        margin = float(margin)
        loss_function = triplet_loss(margin=margin)
    else:
        raise Exception('unknown loss')

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr_init,
        decay_steps=args.lr_drop_step,
        decay_rate=args.lr_drop_value,
        staircase=True)

    model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(lr_schedule))

    with tf.summary.create_file_writer(args.train_dir + "/logs").as_default():
        while True:
            cur_step = model.optimizer.iterations.numpy()
            lr = model.optimizer.lr(cur_step).numpy()
            print('lr', lr)

            history = model.fit(dataset, steps_per_epoch=args.steps_per_epoch)

            cur_step = model.optimizer.iterations.numpy()
            lr = model.optimizer.lr(cur_step).numpy()

            tf.summary.scalar('training/loss', data=history.history['loss'][-1], step=cur_step)
            tf.summary.scalar('training/lr', data=lr, step=cur_step)
            tf.summary.scalar('training/batch_size', data=args.batch_size, step=cur_step)

            save_to = os.path.join(args.train_dir, 'weights-{}'.format(cur_step))
            model.save_weights(save_to)

            print('Saved: {}'.format(save_to))

            if args.dump_embeddings:
                hard_positives, hard_negatives = dump_embeddings(model, dataset, args.train_dir)

                hard_positives = tf.convert_to_tensor(hard_positives)
                hard_negatives = tf.convert_to_tensor(hard_negatives)

                tf.summary.image('hard_positives', hard_positives, cur_step, max_outputs=10)
                tf.summary.image('hard_negatives', hard_negatives, cur_step, max_outputs=10)

            if args.test_images_folder:
                if args.test_gallery_folder:
                    gallery_folder = args.test_gallery_folder
                else:
                    gallery_folder = args.gallery_folder

                top1, top5, top10, mean_pos = test_model(model_path=None, model_backend=None,
                                                         model=model,
                                                         gallery_path=gallery_folder,
                                                         test_data_path=args.test_images_folder,
                                                         test_data_type='crops',
                                                         test_annotation_path=None,
                                                         input_size=args.input_size,
                                                         imshow_delay=-1)

                tf.summary.scalar('test/top1', data=top1, step=cur_step)
                tf.summary.scalar('test/top5', data=top5, step=cur_step)
                tf.summary.scalar('test/top10', data=top10, step=cur_step)
                tf.summary.scalar('test/mean_pos', data=mean_pos, step=cur_step)

            if cur_step > args.max_iters:
                break


if __name__ == '__main__':
    main()
