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
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from text_detection.model import pixel_link_model
from text_detection.evaluation import eval
from text_detection.dataset import get_neighbours, is_valid_coord, TFRecordDataset
from text_detection.common import parse_epoch


def decode_image(segm_scores, link_scores, segm_conf_threshold, link_conf_threshold):
    """ Convert softmax scores to mask. """

    segm_mask = segm_scores >= segm_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    points = list(zip(*np.where(segm_mask)))
    height, width = np.shape(segm_mask)
    group_mask = dict.fromkeys(points, -1)

    def find_parent(point):
        return group_mask[point]

    def set_parent(point, parent):
        group_mask[point] = parent

    def is_root(point):
        return find_parent(point) == -1

    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True

        if update_parent:
            set_parent(point, root)

        return root

    def join(point1, point2):
        root1 = find_root(point1)
        root2 = find_root(point2)

        if root1 != root2:
            set_parent(root1, root2)

    def get_all():
        root_map = {}

        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]

        mask = np.zeros_like(segm_mask, dtype=np.int32)
        for point in points:
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask

    for point in points:
        y_coord, x_coord = point
        neighbours = get_neighbours(x_coord, y_coord)
        for n_idx, (neighbour_x, neighbour_y) in enumerate(neighbours):
            if is_valid_coord(neighbour_x, neighbour_y, width, height):
                link_value = link_mask[y_coord, x_coord, n_idx]
                segm_value = segm_mask[neighbour_y, neighbour_x]
                if link_value and segm_value:
                    join(point, (neighbour_y, neighbour_x))

    mask = get_all()
    return mask


def rect_to_xys(rect, image_shape):
    """ Converts rotated rectangle to points. """

    height, width = image_shape[0:2]

    def get_valid_x(x_coord):
        return np.clip(x_coord, 0, width - 1)

    def get_valid_y(y_coord):
        return np.clip(y_coord, 0, height - 1)

    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x_coord, y_coord) in enumerate(points):
        x_coord = get_valid_x(x_coord)
        y_coord = get_valid_y(y_coord)
        points[i_xy, :] = [x_coord, y_coord]
    points = np.reshape(points, -1)
    return points


def min_area_rect(contour):
    """ Returns minimum area rectangle. """

    (center_x, cencter_y), (width, height), theta = cv2.minAreaRect(contour)
    return [center_x, cencter_y, width, height, theta], width * height


def mask_to_bboxes(mask, config, image_shape):
    """ Converts mask to bounding boxes. """

    image_h, image_w = image_shape[0:2]

    min_area = config['min_area']
    min_height = config['min_height']

    bboxes = []
    max_bbox_idx = mask.max()
    mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

    for bbox_idx in range(1, max_bbox_idx + 1):
        bbox_mask = (mask == bbox_idx).astype(np.uint8)
        cnts = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)

        box_width, box_height = rect[2:-1]
        if min(box_width, box_height) < min_height:
            continue

        if rect_area < min_area:
            continue

        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)

    return bboxes


def decode_batch(segm_scores, link_scores, config):
    """ Returns boxes mask for each input image in batch."""

    batch_size = segm_scores.shape[0]
    batch_mask = []
    for image_idx in range(batch_size):
        image_pos_pixel_scores = segm_scores[image_idx, :, :]
        image_pos_link_scores = link_scores[image_idx, :, :, :]
        mask = decode_image(image_pos_pixel_scores, image_pos_link_scores,
                            config['segm_conf_thr'], config['link_conf_thr'])
        batch_mask.append(mask)
    return np.asarray(batch_mask, np.int32)


def to_boxes(image_data, segm_pos_scores, link_pos_scores, conf):
    """ Returns boxes for each image in batch. """

    mask = decode_batch(segm_pos_scores, link_pos_scores, conf)[0, ...]
    bboxes = mask_to_bboxes(mask, conf, image_data.shape)

    return bboxes


def softmax(logits):
    """ Returns softmax given logits. """

    max_logits = np.max(logits, axis=-1, keepdims=True)
    numerator = np.exp(logits - max_logits)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator


def test(args, config, model=None, dataset=None):
    """This function performs testing of text detection neural network."""

    print('Evaluating:', args.weights)

    config['segm_conf_thr'] = 0.8
    config['link_conf_thr'] = 0.8

    if model is None:
        model = pixel_link_model(tf.keras.Input(shape=list(args.resolution)[::-1] + [3]), config)
    model.load_weights(args.weights)
    if dataset is None:
        dataset, _ = TFRecordDataset(args.dataset, config, test=True)()

    pr_annotations = []
    gt_annotations = []

    for image_tensor, coordinates_tensor, labels_tensor in tqdm(dataset, desc='Evaluation'):
        original_image = image_tensor.numpy().astype(np.float32)

        x_coordinates, y_coordinates = coordinates_tensor.numpy()
        labels = labels_tensor.numpy()
        x_coordinates *= original_image.shape[1]
        y_coordinates *= original_image.shape[0]

        gt_annotation = []
        for i in range(x_coordinates.shape[0]):
            bbox_xs = x_coordinates[i].reshape([-1, 1])
            bbox_ys = y_coordinates[i].reshape([-1, 1])
            points = np.hstack([bbox_xs, bbox_ys]).reshape([-1]).astype(np.int32).tolist()
            gt_annotation.append(
                {'points': points, 'transcription': '###' if labels[i] == -1 else 'GOOD_WORD'})

        gt_annotations.append(gt_annotation)
        image = original_image.astype(np.float32)
        image = cv2.resize(image, tuple(args.resolution))
        segm_logits, link_logits = model.predict(np.array([image]))

        segm_scores = softmax(segm_logits)
        link_scores = softmax(link_logits)
        bboxes = to_boxes(original_image, segm_scores[:, :, :, 1],
                          link_scores[:, :, :, :, 1], config)
        pr_annotations.append(
            [{'points': bbox.reshape([-1]), 'confidence': 1.0} for bbox in bboxes])

        if args.imshow_delay >= 0:
            eval([pr_annotations[-1]], [gt_annotations[-1]],
                 [cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_RGB2BGR)],
                 args.imshow_delay)

    method_recall, method_precision, method_hmean, _ = eval(pr_annotations, gt_annotations)

    epoch = parse_epoch(args.weights)
    ema = 'ema' in os.path.basename(args.weights)

    if ema:
        tf.summary.scalar('ema/hmean', data=method_hmean, step=epoch)
        tf.summary.scalar('ema/precision', data=method_precision, step=epoch)
        tf.summary.scalar('ema/recall', data=method_recall, step=epoch)
    else:
        tf.summary.scalar('common/hmean', data=method_hmean, step=epoch)
        tf.summary.scalar('common/precision', data=method_precision, step=epoch)
        tf.summary.scalar('common/recall', data=method_recall, step=epoch)

    return method_recall, method_precision, method_hmean
