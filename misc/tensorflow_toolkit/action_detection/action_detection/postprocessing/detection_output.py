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

from collections import namedtuple

import numpy as np

from action_detection.postprocessing.metrics import matrix_iou


Detections = namedtuple('Detections', 'loc, scores')
Actions = namedtuple('Detections', 'loc, scores, action_labels, action_scores, id')


def nms(input_bboxes, input_scores, threshold, keep_top_k, min_score=0.01):
    """Carry out default NMS algorithm over the input boxes.

    :param input_bboxes: Input boxes
    :param input_scores: Detection scores of boxes
    :param threshold: Min IoU value to merge boxes
    :param keep_top_k: Max number of boxes to output
    :param min_score: Min score value to output box
    :return: Filtered box IDs
    """

    if len(input_bboxes) == 0:
        return []

    if len(input_bboxes) > keep_top_k:
        indices = np.argsort(-input_scores)[:keep_top_k]
        scores = input_scores[indices]
        bboxes = input_bboxes[indices]
    else:
        scores = np.copy(input_scores)
        indices = np.arange(len(scores))
        bboxes = input_bboxes

    similarity_matrix = matrix_iou(bboxes, bboxes)

    out_ids = []
    for _ in xrange(len(bboxes)):
        bbox_id = np.argmax(scores)
        bbox_score = scores[bbox_id]
        if bbox_score < min_score:
            break

        out_ids.append(indices[bbox_id])
        scores[bbox_id] = 0.0

        iou_values = similarity_matrix[bbox_id]
        scores[iou_values > threshold] = 0.0

    return np.array(out_ids, dtype=np.int32)


def soft_nms(input_bboxes, input_scores, keep_top_k, sigma, min_score):
    """Carry out Soft-NMS algorithm over the input boxes.

    :param input_bboxes: Input boxes
    :param input_scores: Detection scores of boxes
    :param keep_top_k: Max number of boxes to output
    :param sigma: Algorithm parameter
    :param min_score: Min score value to output box
    :return: Filtered box IDs
    """

    if len(input_bboxes) == 0:
        return [], []

    if len(input_bboxes) > keep_top_k:
        indices = np.argsort(-input_scores)[:keep_top_k]
        scores = input_scores[indices]
        bboxes = input_bboxes[indices]
    else:
        scores = np.copy(input_scores)
        indices = np.arange(len(scores))
        bboxes = input_bboxes

    similarity_matrix = matrix_iou(bboxes, bboxes)

    out_ids = []
    out_scores = []
    for _ in xrange(len(bboxes)):
        bbox_id = np.argmax(scores)
        bbox_score = scores[bbox_id]
        if bbox_score < min_score:
            break

        out_ids.append(indices[bbox_id])
        out_scores.append(bbox_score)
        scores[bbox_id] = 0.0

        iou_values = similarity_matrix[bbox_id]
        scores *= np.exp(np.negative(np.square(iou_values) / sigma))

    return np.array(out_ids, dtype=np.int32), np.array(out_scores, dtype=np.float32)


def ssd_detection_output(batch_bboxes, batch_conf, bg_class, min_conf=0.01, out_top_k=200,
                         nms_overlap=0.45, nms_top_k=400):
    """Process network output to translate it into the bboxes with labels.

    :param batch_bboxes: All bboxes
    :param batch_conf: All detection scores
    :param bg_class: ID of background class
    :param min_conf: Min score value to output box
    :param out_top_k: Max number of boxes per image to output
    :param nms_overlap: NMS parameter
    :param nms_top_k: NMS parameter
    :return: List of detections
    """

    assert batch_bboxes.shape[:2] == batch_conf.shape[:2]
    assert batch_bboxes.shape[2] == 4

    num_classes = batch_conf.shape[-1]
    assert num_classes > 1

    all_detections = []
    for sample_id in xrange(batch_bboxes.shape[0]):
        sample_bboxes = batch_bboxes[sample_id]
        sample_conf = batch_conf[sample_id]

        all_sample_detections = []
        for label in xrange(num_classes):
            if label == bg_class:
                continue

            sample_scores = sample_conf[:, label]

            valid_mask = sample_scores > min_conf
            # noinspection PyTypeChecker
            if np.sum(valid_mask) == 0:
                continue

            valid_bboxes = sample_bboxes[valid_mask]
            valid_scores = sample_scores[valid_mask]

            merged_ids = nms(valid_bboxes, valid_scores, nms_overlap, nms_top_k)
            if len(merged_ids) > 0:
                out_bboxes = valid_bboxes[merged_ids].reshape([-1, 4])
                out_scores = valid_scores[merged_ids].reshape([-1])

                for i in xrange(len(out_scores)):
                    all_sample_detections.append((out_bboxes[i], label, out_scores[i]))

        if len(all_sample_detections) > out_top_k:
            all_sample_detections.sort(key=lambda tup: tup[2], reverse=True)
            all_sample_detections = all_sample_detections[:out_top_k]

        sample_detections = {}
        for bbox, label, score in all_sample_detections:
            if label not in sample_detections:
                sample_detections[label] = {'loc': [bbox],
                                            'scores': [score]}
            else:
                last_data = sample_detections[label]
                last_data['loc'].append(bbox)
                last_data['scores'].append(score)

        out_sample_detections = {label: Detections(loc=np.stack(sample_detections[label]['loc']),
                                                   scores=np.stack(sample_detections[label]['scores']))
                                 for label in sample_detections}

        all_detections.append(out_sample_detections)

    return all_detections


def ssd_warp_gt(batch_bboxes, batch_labels, bg_class):
    """Translates Ground truth boxes and labels into the internal format.

    :param batch_bboxes: Bbox coordinates
    :param batch_labels: Bbox labels
    :param bg_class: ID of background label
    :return: List of boxes
    """

    assert batch_bboxes.shape[0] == batch_labels.shape[0]

    all_gt = []
    for sample_id in xrange(batch_bboxes.shape[0]):
        sample_bboxes = batch_bboxes[sample_id]
        sample_labels = batch_labels[sample_id]

        valid_mask = np.logical_and(sample_labels >= 0, sample_labels != bg_class)
        if np.sum(valid_mask) == 0:
            all_gt.append([])
            continue

        valid_bboxes = sample_bboxes[valid_mask]
        valid_labels = sample_labels[valid_mask]

        unique_labels = np.unique(valid_labels)
        sample_detections = {}
        for label in unique_labels:
            label_mask = valid_labels == label
            class_bboxes = valid_bboxes[label_mask]

            sample_detections[label] = Detections(loc=class_bboxes, scores=None)

        all_gt.append(sample_detections)

    return all_gt


def action_detection_output(batch_bboxes, batch_det_conf, batch_action_conf, bg_class,
                            min_det_conf=0.01, min_action_conf=0.01, out_top_k=400,
                            nms_top_k=400, nms_sigma=0.6, do_nms=True):
    """Process network output to translate it into the bboxes with detection scores and action labels.

    :param batch_bboxes: All bboxes
    :param batch_det_conf: All detection scores
    :param batch_action_conf: All action scores
    :param bg_class: ID of background class
    :param min_det_conf: Min score value to output box
    :param min_action_conf: Min score value for action confidence
    :param out_top_k: Max number of boxes per image to output
    :param nms_top_k: NMS parameter
    :param nms_sigma: NMS parameter
    :param do_nms: Whether to run NMS algorithm
    :return: List of detections
    """

    assert batch_bboxes.shape[:2] == batch_det_conf.shape[:2]
    assert batch_bboxes.shape[:2] == batch_action_conf.shape[:2]
    assert batch_bboxes.shape[2] == 4

    num_det_classes = batch_det_conf.shape[-1]
    assert num_det_classes == 2

    num_action_classes = batch_action_conf.shape[-1]
    assert num_action_classes > 1

    det_class = (bg_class + 1) % 2

    all_detections = []
    for sample_id in xrange(batch_bboxes.shape[0]):
        sample_bboxes = batch_bboxes[sample_id]
        sample_det_scores = batch_det_conf[sample_id, :, det_class]
        sample_action_conf = batch_action_conf[sample_id]

        valid_mask = sample_det_scores > min_det_conf
        # noinspection PyTypeChecker
        if np.sum(valid_mask) == 0:
            all_detections.append({det_class: []})
            continue

        valid_bboxes = sample_bboxes[valid_mask]
        valid_det_scores = sample_det_scores[valid_mask]
        valid_det_conf = sample_action_conf[valid_mask]

        if do_nms:
            filtered_ids, filtered_scores = soft_nms(valid_bboxes, valid_det_scores, nms_top_k, nms_sigma, min_det_conf)
        else:
            filtered_scores = np.copy(valid_det_scores)
            filtered_ids = np.argsort(-filtered_scores)

        if len(filtered_ids) > 0:
            out_bboxes = valid_bboxes[filtered_ids].reshape([-1, 4])
            out_det_scores = filtered_scores.reshape([-1])
            out_action_conf = valid_det_conf[filtered_ids].reshape([-1, num_action_classes])

            if 0 < out_top_k < len(out_det_scores):
                out_bboxes = out_bboxes[:out_top_k]
                out_det_scores = out_det_scores[:out_top_k]
                out_action_conf = out_action_conf[:out_top_k]

            out_action_label = np.argmax(out_action_conf, axis=-1)
            out_action_score = np.max(out_action_conf, axis=-1)

            if min_action_conf is not None and min_action_conf > 0.0:
                out_action_label[out_action_score < min_action_conf] = 0

            sample_detections = Actions(loc=out_bboxes,
                                        scores=out_det_scores,
                                        action_labels=out_action_label,
                                        action_scores=out_action_score,
                                        id=None)
            all_detections.append({det_class: sample_detections})
        else:
            all_detections.append({det_class: []})
            continue

    return all_detections


def action_warp_gt(batch_bboxes, batch_labels, bg_class, batch_track_ids=None):
    """Translates Ground truth boxes and actions into the internal format.

    :param batch_bboxes: Bbox coordinates
    :param batch_labels: Bbox labels
    :param bg_class: ID of background label
    :param batch_track_ids: ID of track in a batch
    :return: List of boxes
    """

    assert batch_bboxes.shape[0] == batch_labels.shape[0]

    det_class = (bg_class + 1) % 2

    all_gt = []
    for sample_id in xrange(batch_bboxes.shape[0]):
        sample_bboxes = batch_bboxes[sample_id]
        sample_labels = batch_labels[sample_id]
        sample_track_ids = batch_track_ids[sample_id] if batch_track_ids is not None else None

        valid_mask = sample_labels >= 0
        # noinspection PyTypeChecker
        if np.sum(valid_mask) == 0:
            all_gt.append([])
            continue

        valid_bboxes = sample_bboxes[valid_mask]
        valid_labels = sample_labels[valid_mask]
        valid_track_ids = sample_track_ids[valid_mask] if sample_track_ids is not None else None

        sample_detections = {det_class: Actions(loc=valid_bboxes,
                                                scores=None,
                                                action_labels=valid_labels,
                                                action_scores=None,
                                                id=valid_track_ids)}

        all_gt.append(sample_detections)

    return all_gt
