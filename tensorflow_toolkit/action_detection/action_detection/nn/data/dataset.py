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

from __future__ import print_function


from functools import partial
from os.path import exists

import json
import numpy as np
import tensorflow as tf
from tqdm import trange

from action_detection.nn.data.core import parse_text_records, decode_jpeg, encode_image


def get_classification_dataset(data_file_path, num_classes, image_size, batch_size, do_shuffle, name,
                               prefetch_size=1, num_threads=5, process_fn=None):
    """Prepares classification dataset object.

    :param data_file_path: Path to file with rows: [image_path label]
    :param num_classes: Number of classification classes
    :param image_size: Target image size
    :param batch_size: Size of batch
    :param do_shuffle: Whether shuffle records
    :param name: Name of dataset
    :param prefetch_size: Size of prefetch queue
    :param num_threads: Number of threads to load images
    :param process_fn: Function to transform each image
    :return: Classification dataset
    """

    def _convert_label(input_value):
        """Checks if input label valid and returns int value

        :param input_value: Input value
        :return: Int value if valid and None else
        """

        int_x = int(input_value)
        return int_x if 0 <= int_x < num_classes else None

    def _mapping_fn(filename, label):
        """Loads image by the specified file path.

        :param filename: Path to image
        :param label: Image label
        :return: Image and label tuple
        """

        image = decode_jpeg(filename, image_size.c)

        if process_fn is not None:
            image = process_fn(image)

        image_blob = encode_image(image, image_size.h, image_size.w, True, 255.0)

        return image_blob, label

    image_paths, labels = parse_text_records(data_file_path, types=['path', _convert_label])

    with tf.name_scope(name):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        if do_shuffle:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(len(image_paths), count=-1))
        else:
            dataset = dataset.repeat(count=-1)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(_mapping_fn, batch_size, num_threads,
                                                                   drop_remainder=True))

        if prefetch_size is not None and prefetch_size > 0:
            dataset = dataset.prefetch(prefetch_size)

        return dataset


class DetectionClassBalancingGenerator(object):
    """Functor which allow to load balanced data according bbox labels.
    """

    def __init__(self, image_paths, annotation_paths, labels_map=None, ignore_labels=None, min_class_queue_size=1):
        """Constructor

        :param image_paths: List of image paths
        :param annotation_paths: List of annotation paths
        :param labels_map: Map of input labels
        :param ignore_labels: List of ignored labels
        :param min_class_queue_size: Minimal size of queue
        """

        assert len(image_paths) == len(annotation_paths)
        assert min_class_queue_size >= 1

        class_queues = {}
        for sample_id in trange(len(image_paths), desc='Reading data'):
            annotation_path = annotation_paths[sample_id]

            with open(annotation_path, 'r') as read_file:
                annotation_data = json.load(read_file)

            labels = [b['label'] for b in annotation_data]
            labels = np.unique(labels)
            if labels_map is not None:
                labels = [labels_map[l] for l in labels]

            if ignore_labels is not None:
                labels = [l for l in labels if l not in ignore_labels]

            for label in labels:
                if label not in class_queues:
                    class_queues[label] = []

                class_queues[label].append(sample_id)

        for label in class_queues:
            if len(class_queues[label]) < min_class_queue_size:
                raise Exception('Error: Cannot find frames with {} label!'.format(label))
        print('Info: loaded {} frames with {} labels'.format(len(image_paths), len(class_queues)))

        min_queue_size = np.min([len(class_queues[label]) for label in class_queues])
        print('Info: min class queue size: {}'.format(min_queue_size))

        self._image_paths = image_paths
        self._annotation_paths = annotation_paths
        self._class_queues = class_queues
        self._min_queue_size = min_queue_size

    def __call__(self):
        """Generates next portion of data.

        :return: Iterator object
        """

        class_subsets = []
        for label in self._class_queues:
            class_frame_ids = np.copy(self._class_queues[label]).astype(np.int32)
            subset_ids = np.random.choice(class_frame_ids, self._min_queue_size, replace=False)
            class_subsets.append(subset_ids)

        final_num_classes = len(class_subsets)
        final_ids = np.zeros([final_num_classes * self._min_queue_size], dtype=np.int32)
        for i in range(final_num_classes):
            final_ids[i::final_num_classes] = class_subsets[i]
        np.random.shuffle(final_ids)

        out_image_paths = []
        out_annotation_paths = []
        for sample_id in final_ids:
            out_image_paths.append(self._image_paths[sample_id])
            out_annotation_paths.append(self._annotation_paths[sample_id])

        yield out_image_paths, out_annotation_paths

    def get_chunk_size(self):
        """Returns size of prepared chunk of data.

        :return: Number of prepared images.
        """

        return len(self._class_queues) * self._min_queue_size


def get_detection_dataset(data_file_path, image_size, batch_size, do_shuffle, name, prefetch_size=1, num_threads=5,
                          image_process_fn=None, tuple_process_fn=None, max_num_objects_per_image=None, cache=False,
                          use_difficult=True, labels_map=None, output_labels_stat=False, ignore_classes=None,
                          use_class_balancing=False):
    """Prepares detection dataset object.

    :param data_file_path: Path to file with data
    :param image_size: Image size
    :param batch_size: Size of batch
    :param do_shuffle: Whether shuffle output records
    :param name: Name of dataset
    :param prefetch_size: Size of prefetch queue
    :param num_threads: Num threads to load image
    :param image_process_fn: Function to process image independently
    :param tuple_process_fn: Function to process image and annotation simultaneously
    :param max_num_objects_per_image: Max number of boxes on image
    :param cache: Whether to enable dataset caching
    :param use_difficult: Whether to include difficult samples
    :param labels_map: Map of input labels
    :param output_labels_stat: Whether to output label stats
    :param ignore_classes: List of ignores classes
    :param use_class_balancing: Enable class balancing scheme
    :return: Detection dataset
    """

    def _get_max_num_objects(data_path):
        """Returns mux number of boxes on single image.

        :param data_path: Path to file with data
        :return: Number of boxes
        """

        max_num_bboxes = 0
        with open(data_path, 'r') as input_stream:
            for line in input_stream:
                if line.endswith('\n'):
                    line = line[:-len('\n')]

                if len(line) == 0:
                    continue

                line_data = line.split(' ')
                assert len(line_data) == 2

                annot_path = line_data[-1]
                if exists(annot_path):
                    with open(annot_path, 'r') as read_file:
                        bboxes = json.load(read_file)

                    max_num_bboxes = np.maximum(max_num_bboxes, len(bboxes))

        return max_num_bboxes

    def _get_labels_stat(data_path):
        """Returns label stats.

        :param data_path: Path to file with data
        :return: Dictionary with counts of each label
        """

        all_labels = []
        with open(data_path, 'r') as input_stream:
            for line in input_stream:
                if line.endswith('\n'):
                    line = line[:-len('\n')]

                if len(line) == 0:
                    continue

                line_data = line.split(' ')
                assert len(line_data) == 2

                annot_path = line_data[-1]
                if exists(annot_path):
                    with open(annot_path, 'r') as read_file:
                        bboxes = json.load(read_file)

                    if labels_map is not None:
                        new_labels = [labels_map[b['label']] for b in bboxes]
                    else:
                        new_labels = [b['label'] for b in bboxes]

                    all_labels.extend(new_labels)

        unique_labels, unique_counts = np.unique(all_labels, return_counts=True)
        out_counts = {l: c for l, c in zip(unique_labels, unique_counts)}

        return out_counts

    def _read_annot(annot_path, max_num_objects):
        """Loads annotation from specified file.

        :param annot_path: Path to annotation file
        :param max_num_objects: Max number of boxes on single image
        :return: Tensor with annotation
        """

        def _py_internal_fn(trg_path):
            with open(trg_path, 'r') as read_file:
                bboxes = json.load(read_file)

            out_labels = []
            out_bboxes = []
            for bbox in bboxes:
                if not use_difficult and 'difficult' in bbox and bbox['difficult']:
                    continue

                decoded_label = labels_map[bbox['label']] if labels_map is not None else bbox['label']
                out_labels.append(decoded_label)

                out_bboxes.append([bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax']])

            assert len(out_labels) == len(out_bboxes)

            out_labels = np.array(out_labels, dtype=np.int32)
            out_bboxes = np.array(out_bboxes, dtype=np.float32)

            if out_bboxes.shape[0] < max_num_objects:
                pad_size = max_num_objects - out_bboxes.shape[0]
                out_labels = np.pad(out_labels, (0, pad_size), 'constant', constant_values=-1)
                out_bboxes = np.pad(out_bboxes, ((0, pad_size), (0, 0)), 'constant', constant_values=0.0)
            elif out_bboxes.shape[0] > max_num_objects:
                ids_subset = np.random.choice(np.arange(start=0, stop=len(bboxes)), max_num_objects, replace=False)
                out_labels = out_labels[ids_subset]
                out_bboxes = out_bboxes[ids_subset]

            return out_labels, out_bboxes

        return tf.py_func(_py_internal_fn, [annot_path], [tf.int32, tf.float32])

    def _unpack_data(image_path, annot_path, size):
        """Loads image and annotation from files.

        :param image_path: Path to image
        :param annot_path: Path to annotation file
        :param size: Max number of boxes on single image
        :return: Image and annotation tensors
        """

        image = decode_jpeg(image_path, image_size.c)
        labels, bboxes = _read_annot(annot_path, size)

        labels.set_shape([size])
        bboxes.set_shape([size, 4])

        if tuple_process_fn is not None:
            image, labels, bboxes = tuple_process_fn(image, labels, bboxes)

        if image_process_fn is not None:
            image = image_process_fn(image)

        encoded_image = encode_image(image, image_size.h, image_size.w, True, 255.0)

        return encoded_image, labels, bboxes

    if max_num_objects_per_image is None:
        max_num_objects_per_image = _get_max_num_objects(data_file_path)

    image_paths, annot_paths = parse_text_records(data_file_path, types=['path', 'path'])

    with tf.name_scope(name):
        if use_class_balancing:
            data_generator = DetectionClassBalancingGenerator(image_paths, annot_paths, labels_map, ignore_classes)
            data_chunk_size = data_generator.get_chunk_size()

            chunk_dataset = tf.data.Dataset.from_generator(data_generator, (tf.string, tf.string),
                                                           (tf.TensorShape([data_chunk_size]),
                                                            tf.TensorShape([data_chunk_size])))
            chunk_dataset = chunk_dataset.repeat(count=-1)
            chunk_iterator = chunk_dataset.make_one_shot_iterator()
            image_paths_chunk, annot_paths_chunk = chunk_iterator.get_next()

            dataset = tf.data.Dataset.from_tensor_slices((image_paths_chunk, annot_paths_chunk))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, annot_paths))
            data_chunk_size = len(image_paths)

        if do_shuffle:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(data_chunk_size, count=-1))
        else:
            dataset = dataset.repeat(count=-1)

        mapping_fn = partial(_unpack_data, size=max_num_objects_per_image)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(mapping_fn, batch_size, num_threads,
                                                                   drop_remainder=True))

        if prefetch_size is not None and prefetch_size > 0:
            dataset = dataset.prefetch(prefetch_size)

        if cache:
            dataset = dataset.cache()

    if output_labels_stat:
        label_counts = _get_labels_stat(data_file_path)
        return dataset, label_counts
    else:
        return dataset
