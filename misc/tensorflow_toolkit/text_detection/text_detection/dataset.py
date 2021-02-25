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

""" This module contains TFRecordDataset class. """

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import math_ops

import cv2


def points_to_contour(points):
    """ Converts points to contour. """

    return np.asarray([[list(point)] for point in points], dtype=np.int32)


def points_to_contours(points):
    """ Converts points to contours. """

    return np.asarray([points_to_contour(points)])


def get_neighbours(x_coord, y_coord):
    """ Returns 8-point neighbourhood of given point. """

    return [(x_coord - 1, y_coord - 1), (x_coord, y_coord - 1), (x_coord + 1, y_coord - 1), \
            (x_coord - 1, y_coord), (x_coord + 1, y_coord), \
            (x_coord - 1, y_coord + 1), (x_coord, y_coord + 1), (x_coord + 1, y_coord + 1)]


def is_valid_coord(x_coord, y_coord, width, height):
    """ Returns true if given point inside image frame. """

    return 0 <= x_coord < width and 0 <= y_coord < height


def tf_min_area_rect(x_coords, y_coords):
    """ Returns rotated rectangles for given set of points. """

    def min_area_rect(x_coords, y_coords):
        num_rects = x_coords.shape[0]
        box = np.empty((num_rects, 5), dtype=np.float32)
        for idx in range(num_rects):
            points = zip(x_coords[idx, :], y_coords[idx, :])
            (center_x, center_y), (width, height), theta = \
                cv2.minAreaRect(points_to_contour(points))
            box[idx, :] = [center_x, center_y, width, height, theta]
        return box

    rects = tf.numpy_function(min_area_rect, [x_coords, y_coords], x_coords.dtype)
    rects.set_shape([None, 5])
    return rects


def safe_divide(numerator, denominator, name):
    """ Performs safe division. """

    return tf.where(
        math_ops.greater(denominator, 0),
        math_ops.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)


def bboxes_intersection(bbox_ref, bboxes):
    """ Computes bounding boxes intersection. """

    bboxes = tf.transpose(a=bboxes)
    bbox_ref = tf.transpose(a=bbox_ref)
    int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
    int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
    int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
    int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
    height = tf.maximum(int_ymax - int_ymin, 0.)
    widths = tf.maximum(int_xmax - int_xmin, 0.)
    inter_vol = height * widths
    bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
    scores = safe_divide(inter_vol, bboxes_vol, 'intersection')
    return scores


def random_rotate90(image, bboxes, x_coords, y_coords):
    """ Randomly rotate image and bounding boxes by 0, 90, 180, 270 degrees. """
    rotate_by_90_k_times = tf.random.uniform([], maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=rotate_by_90_k_times)
    bboxes, x_coords, y_coords = rotate90(bboxes, x_coords, y_coords, rotate_by_90_k_times)
    return image, bboxes, x_coords, y_coords


def rotate_point_by_90(x_coord, y_coord, rotate_by_90_k_times):
    """ Rotate point by 90 degrees (clockwise). """

    cos = tf.constant([1.0, 0.0, -1.0, 0.0])
    sin = tf.constant([0.0, -1.0, 0.0, 1.0])

    x1_coord = x_coord - 0.5
    y1_coord = y_coord - 0.5

    x_coord = x1_coord * cos[rotate_by_90_k_times] - y1_coord * sin[rotate_by_90_k_times] + 0.5
    y_coord = x1_coord * sin[rotate_by_90_k_times] + y1_coord * cos[rotate_by_90_k_times] + 0.5

    return x_coord, y_coord


def rotate90(bboxes, x_coords, y_coords, k):
    """ Rotate bounding boxes by 90 degrees."""

    ymin, xmin, ymax, xmax = [bboxes[:, i] for i in range(4)]
    xmin, ymin = rotate_point_by_90(xmin, ymin, k)
    xmax, ymax = rotate_point_by_90(xmax, ymax, k)

    new_xmin = tf.minimum(xmin, xmax)
    new_xmax = tf.maximum(xmin, xmax)

    new_ymin = tf.minimum(ymin, ymax)
    new_ymax = tf.maximum(ymin, ymax)

    bboxes = tf.stack([new_ymin, new_xmin, new_ymax, new_xmax])
    bboxes = tf.transpose(a=bboxes)

    x_coords, y_coords = rotate_point_by_90(x_coords, y_coords, k)
    return bboxes, x_coords, y_coords


def bboxes_resize(bbox_ref, bboxes, x_coords, y_coords):
    """ Resize bounding box relatively to reference bounding box. """

    h_ref = bbox_ref[2] - bbox_ref[0]
    w_ref = bbox_ref[3] - bbox_ref[1]

    bboxes = bboxes - tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
    x_coords = x_coords - bbox_ref[1]
    y_coords = y_coords - bbox_ref[0]

    bboxes = bboxes / tf.stack([h_ref, w_ref, h_ref, w_ref])
    x_coords = x_coords / w_ref
    y_coords = y_coords / h_ref

    return bboxes, x_coords, y_coords


def tf_prepare_groundtruth_for_image(x_coords, y_coords, labels, config):
    """ Generate groundtruth data for given image. """

    height, width = config['score_map_shape']
    num_neighbours = config['num_neighbours']

    def prepare_groundtruth_for_image(normed_xs, normed_ys, labels):


        num_positive_bboxes = np.sum(np.asarray(labels) == config['text_label'])
        x_coords = normed_xs * width
        y_coords = normed_ys * height

        mask = np.zeros([height, width], np.int32)
        segm_labels = np.ones([height, width], np.int32) * config['background_label']
        segm_weights = np.zeros([height, width], np.float32)

        link_labels = np.zeros([height, width, num_neighbours], np.int32)
        link_weights = np.ones([height, width, num_neighbours], np.float32)

        bbox_masks = []
        pos_mask = mask.copy()
        for bbox_idx, (bbox_xs, bbox_ys) in enumerate(zip(x_coords, y_coords)):
            if labels[bbox_idx] == config['background_label']:
                continue

            bbox_mask = mask.copy()
            bbox_contours = points_to_contours(zip(bbox_xs, bbox_ys))
            cv2.drawContours(bbox_mask, bbox_contours, -1, 1, -1)

            bbox_masks.append(bbox_mask)

            if labels[bbox_idx] == config['text_label']:
                pos_mask += bbox_mask

        pos_mask = np.asarray(pos_mask == 1, dtype=np.int32)
        num_positive_pixels = np.sum(pos_mask)

        sum_mask = np.sum(bbox_masks, axis=0)
        not_overlapped_mask = sum_mask == 1

        for bbox_idx, bbox_mask in enumerate(bbox_masks):
            bbox_label = labels[bbox_idx]
            if bbox_label == config['ignore_label']:
                segm_labels += bbox_mask * not_overlapped_mask * config['ignore_label']
                continue

            if labels[bbox_idx] == config['background_label']:
                continue

            text_boxes_mask = bbox_mask * pos_mask
            segm_labels += text_boxes_mask * bbox_label

            num_bbox_pixels = np.sum(text_boxes_mask)
            if num_bbox_pixels > 0:
                per_bbox_weight = num_positive_pixels * 1.0 / num_positive_bboxes
                per_pixel_weight = per_bbox_weight / num_bbox_pixels
                segm_weights += text_boxes_mask * per_pixel_weight

            bbox_point_cords = np.where(text_boxes_mask)
            link_labels[bbox_point_cords] = 1

            new_bbox_contours = cv2.findContours(text_boxes_mask.astype(np.uint8), cv2.RETR_CCOMP,
                                                 cv2.CHAIN_APPROX_SIMPLE)[-2]

            text_boxes_border_mask = mask.copy()
            cv2.drawContours(text_boxes_border_mask, new_bbox_contours, -1, 1, 3)
            text_boxes_border_mask *= text_boxes_mask

            border_points = zip(*np.where(text_boxes_border_mask))

            def in_bbox(neighbour_x, neighbour_y):
                return text_boxes_mask[neighbour_y, neighbour_x]

            for y_coord, x_coord in border_points:
                neighbours = get_neighbours(x_coord, y_coord)
                for n_idx, (neighbour_x, neighbour_y) in enumerate(neighbours):
                    if not is_valid_coord(neighbour_x, neighbour_y, width, height) \
                            or not in_bbox(neighbour_x, neighbour_y):
                        link_labels[y_coord, x_coord, n_idx] = 0

        link_weights *= np.expand_dims(segm_weights, axis=-1)

        return segm_labels, segm_weights, link_labels, link_weights

    segm_labels, segm_weights, link_labels, link_weights = \
        tf.numpy_function(prepare_groundtruth_for_image,
                          [x_coords, y_coords, labels],
                          [tf.int32, tf.float32, tf.int32, tf.float32]
                          )

    segm_labels.set_shape([height, width])
    segm_weights.set_shape([height, width])
    link_labels.set_shape([height, width, num_neighbours])
    link_weights.set_shape([height, width, num_neighbours])

    return segm_labels, segm_weights, link_labels, link_weights


def bboxes_filter_overlap(labels, bboxes, x_coords, y_coords, config):
    """ Filter boxes with inappropriate size."""

    scores = bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype), bboxes)

    threshold = config['bbox_crop_overlap']
    assign_value = config['ignore_label']
    if assign_value is not None:
        mask = scores < threshold
        mask = tf.logical_and(mask, tf.equal(labels, config['text_label']))
        labels = tf.where(mask, tf.ones_like(labels) * assign_value, labels)
    else:
        mask = scores > threshold
        labels = tf.boolean_mask(tensor=labels, mask=mask)
        bboxes = tf.boolean_mask(tensor=bboxes, mask=mask)
        x_coords = tf.boolean_mask(tensor=x_coords, mask=mask)
        y_coords = tf.boolean_mask(tensor=y_coords, mask=mask)
    return labels, bboxes, x_coords, y_coords


class TFRecordDataset:
    """ Dataset that is used in training. """

    def __init__(self, path, config, test=False):
        self.config = config
        self.test = test

        dataset = tf.data.TFRecordDataset(path)
        self.size = sum(1 for _ in dataset)

        if not self.test:
            dataset = dataset.shuffle(1000 * self.config['batch_size'])

        dataset = dataset.map(self.parse_tf_record,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if not self.test:
            dataset = dataset.map(self.augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(self.preprocess_input_train)
            dataset = dataset.map(self.prepare_groundtruth,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(self.config['batch_size'] * self.config['num_replicas'],
                                    drop_remainder=True)
            dataset = dataset.repeat()
        else:
            dataset = dataset.map(self.preprocess_input_test)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        self.dataset = dataset

    def __call__(self):
        return self.dataset, self.size

    def parse_tf_record(self, example_proto):
        """ Parses tf record. """

        keys_to_features = {
            'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/shape': tf.io.FixedLenFeature([3], tf.int64),
            'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/x1': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/x2': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/x3': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/x4': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/y1': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/y2': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/y3': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/y4': tf.io.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.io.VarLenFeature(dtype=tf.int64),
        }

        parsed_features = tf.io.parse_single_example(serialized=example_proto,
                                                     features=keys_to_features)

        image = tf.image.decode_jpeg(parsed_features['image/encoded'])
        glabel = tf.sparse.to_dense(parsed_features['image/object/bbox/label'])
        gbboxes = tf.transpose(
            a=tf.stack([tf.sparse.to_dense(parsed_features['image/object/bbox/ymin']),
                        tf.sparse.to_dense(parsed_features['image/object/bbox/xmin']),
                        tf.sparse.to_dense(parsed_features['image/object/bbox/ymax']),
                        tf.sparse.to_dense(parsed_features['image/object/bbox/xmax'])]))

        x1_coord = tf.sparse.to_dense(parsed_features['image/object/bbox/x1'])
        x2_coord = tf.sparse.to_dense(parsed_features['image/object/bbox/x2'])
        x3_coord = tf.sparse.to_dense(parsed_features['image/object/bbox/x3'])
        x4_coord = tf.sparse.to_dense(parsed_features['image/object/bbox/x4'])
        y1_coord = tf.sparse.to_dense(parsed_features['image/object/bbox/y1'])
        y2_coord = tf.sparse.to_dense(parsed_features['image/object/bbox/y2'])
        y3_coord = tf.sparse.to_dense(parsed_features['image/object/bbox/y3'])
        y4_coord = tf.sparse.to_dense(parsed_features['image/object/bbox/y4'])

        x_coords = tf.transpose(a=tf.stack([x1_coord, x2_coord, x3_coord, x4_coord]))
        y_coords = tf.transpose(a=tf.stack([y1_coord, y2_coord, y3_coord, y4_coord]))

        image.set_shape([None, None, 3])

        if self.test:
            stacked = tf.stack([x_coords, y_coords])
            return image, stacked, glabel

        return image, glabel, gbboxes, x_coords, y_coords

    def preprocess_input(self, image):
        image = tf.cast(image, tf.float32)
        shape = image.shape

        def preprocess(image):
            if self.config['model_type'] in ['ka_resnet50', 'ka_vgg16']:
                return tf.keras.applications.resnet50.preprocess_input(image)
            elif self.config['model_type'] in ['ka_mobilenet_v2_1_0', 'ka_mobilenet_v2_1_4']:
                return tf.keras.applications.mobilenet_v2.preprocess_input(image)
            elif self.config['model_type'] in ['mobilenet_v2_ext']:
                return image
            elif self.config['model_type'] in ['ka_xception']:
                return tf.keras.applications.xception.preprocess_input(image)
            else:
                raise Exception('model_type is not specified.')

        image = tf.numpy_function(preprocess, [image], image.dtype)
        image.set_shape(shape)

        return image

    def preprocess_input_train(self, image, labels, x_coords, y_coords):
        return self.preprocess_input(image), labels, x_coords, y_coords

    def preprocess_input_test(self, image, stacked, glabel):
        return self.preprocess_input(image), stacked, glabel

    def augment(self, image, labels, bboxes, x_coords, y_coords):
        """ Augments dataset sample. """

        assert image.dtype == tf.uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if self.config['rotate']:
            image, bboxes, x_coords, y_coords = self.rotate(image, bboxes, x_coords, y_coords)
        if self.config['random_crop']:
            image, labels, bboxes, x_coords, y_coords = self.random_crop(image, labels, bboxes,
                                                                         x_coords, y_coords)
        image = self.resize(image)
        if self.config['using_shorter_side_filtering']:
            labels, bboxes, x_coords, y_coords = self.bboxes_filter_by_shorter_side(labels, bboxes,
                                                                                    x_coords,
                                                                                    y_coords)
        if self.config['distort_color']:
            image = self.distort_color(image)
        image = image * 255.
        return image, labels, x_coords, y_coords

    @tf.function
    def rotate(self, image, bboxes, x_coords, y_coords):
        """ Randomly rotates image and boxes. """

        rnd = tf.random.uniform((), minval=0, maxval=1)
        if tf.less(rnd, self.config['rotation_prob']):
            image, bboxes, x_coords, y_coords = random_rotate90(image, bboxes, x_coords, y_coords)
        return image, bboxes, x_coords, y_coords

    @tf.function
    def random_crop(self, image, labels, bboxes, x_coords, y_coords, max_attempts=200):
        """ Makes a random crop. """

        num_bboxes = tf.shape(input=bboxes)[0]

        if tf.equal(num_bboxes, 0):
            xmin = tf.random.uniform((1, 1), minval=0, maxval=0.9)
            ymin = tf.random.uniform((1, 1), minval=0, maxval=0.9)
            xmax = xmin + tf.constant(0.1, dtype=tf.float32)
            ymax = ymin + tf.constant(0.1, dtype=tf.float32)
            bboxes = tf.concat([ymin, xmin, ymax, xmax], axis=1)
            labels = tf.constant([self.config['background_label']], dtype=tf.int64)
            x_coords = tf.concat([xmin, xmax, xmax, xmin], axis=1)
            y_coords = tf.concat([ymin, ymin, ymax, ymax], axis=1)

        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(input=image),
            bounding_boxes=tf.expand_dims(bboxes, 0),
            min_object_covered=self.config['min_object_covered'],
            aspect_ratio_range=self.config['crop_aspect_ratio_range'],
            area_range=self.config['area_range'],
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0]

        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image.set_shape([None, None, 3])
        bboxes, x_coords, y_coords = bboxes_resize(distort_bbox, bboxes, x_coords, y_coords)
        labels, bboxes, x_coords, y_coords = bboxes_filter_overlap(labels, bboxes, x_coords,
                                                                   y_coords, self.config)
        return cropped_image, labels, bboxes, x_coords, y_coords

    def resize(self, image):
        """ Resizes an image. """

        return tf.squeeze(
            tf.image.resize(tf.expand_dims(image, 0), self.config['train_image_shape']))

    def bboxes_filter_by_shorter_side(self, labels, bboxes, x_coords, y_coords):
        """ Filter boxes by their shorter side."""

        min_height = self.config['min_shorter_side']
        max_height = self.config['max_shorter_side']
        assign_value = self.config['ignore_label']

        x_coords = x_coords * self.config['train_image_shape'][1]
        y_coords = y_coords * self.config['train_image_shape'][0]

        bbox_rects = tf_min_area_rect(x_coords, y_coords)
        widths, heights = bbox_rects[:, 2], bbox_rects[:, 3]
        shorter_sides = tf.minimum(widths, heights)
        if assign_value is not None:
            mask = tf.logical_or(shorter_sides < min_height, shorter_sides > max_height)
            mask = tf.logical_and(mask, tf.equal(labels, self.config['text_label']))
            labels = tf.where(mask, tf.ones_like(labels) * assign_value, labels)
        else:
            mask = tf.logical_and(shorter_sides >= min_height, shorter_sides <= max_height)
            labels = tf.boolean_mask(tensor=labels, mask=mask)
            bboxes = tf.boolean_mask(tensor=bboxes, mask=mask)
            x_coords = tf.boolean_mask(tensor=x_coords, mask=mask)
            y_coords = tf.boolean_mask(tensor=y_coords, mask=mask)
        x_coords = x_coords / self.config['train_image_shape'][1]
        y_coords = y_coords / self.config['train_image_shape'][0]
        return labels, bboxes, x_coords, y_coords

    @tf.function
    def distort_color(self, image):
        """ Distorts color. """

        color_ordering = tf.random.uniform([], maxval=4, dtype=tf.int32)
        if tf.equal(color_ordering, 0):
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif tf.equal(color_ordering, 1):
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        elif tf.equal(color_ordering, 2):
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif tf.equal(color_ordering, 3):
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)

        return tf.clip_by_value(image, 0.0, 1.0)

    def prepare_groundtruth(self, image, labels, x_coords, y_coords):
        """ Prepares groundtruth. """

        segm_labels, segm_weights, link_labels, link_weights = \
            tf_prepare_groundtruth_for_image(x_coords, y_coords, labels, self.config)

        segm_labels = tf.expand_dims(segm_labels, axis=-1)
        segm_weights = tf.expand_dims(segm_weights, axis=-1)

        segm_labels = tf.cast(segm_labels, tf.float32)
        link_labels = tf.cast(link_labels, tf.float32)

        cls_output = tf.keras.layers.concatenate([segm_labels, segm_weights])
        output = tf.keras.layers.concatenate([cls_output, link_labels, link_weights])

        return image, (cls_output, tf.expand_dims(output, axis=-1))
