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

# pylint: disable=invalid-name

from bisect import bisect
from collections import namedtuple

import numpy as np
from tqdm import trange

from action_detection.postprocessing.metrics import iou, matrix_iou


MatchDesc = namedtuple('MatchDesc', 'gt_bbox, gt_label, pred_bbox, pred_label')


def calc_map_mr(predictions, gt, min_iou=0.5, fppi_level=0.1, return_all=False):
    """Calculates mAP and miss-rate metrics.

    :param predictions: Predicted boxes
    :param gt: Ground truth boxes
    :param min_iou: Min IoU value to match GT box
    :param fppi_level: FPPI level to calculate miss-rate metric
    :param return_all: Whether to return metrics by class
    :return: List of metric values
    """

    def _match():
        total_num_candidates = 0
        total_num_matches = 0
        out_matches = {}

        for sample_id in trange(len(predictions), desc='Matching'):
            sample_predictions = predictions[sample_id]
            sample_gt = gt[sample_id]

            predicted_labels = list(sample_predictions)
            gt_labels = list(sample_gt)

            for label in predicted_labels:
                if label in gt_labels:
                    predicted_label_data = sample_predictions[label]
                    gt_bboxes = sample_gt[label].loc

                    sorted_ind = np.argsort(-predicted_label_data.scores)
                    predicted_bboxes = predicted_label_data.loc[sorted_ind]
                    predicted_scores = predicted_label_data.scores[sorted_ind]

                    similarity_matrix = matrix_iou(predicted_bboxes, gt_bboxes)

                    matches = []
                    visited_gt = np.zeros(gt_bboxes.shape[0], dtype=np.bool)
                    for predicted_id in xrange(predicted_bboxes.shape[0]):

                        best_overlap = 0.0
                        best_gt_id = -1
                        for gt_id in xrange(gt_bboxes.shape[0]):
                            if visited_gt[gt_id]:
                                continue

                            overlap_value = similarity_matrix[predicted_id, gt_id]
                            if overlap_value > best_overlap:
                                best_overlap = overlap_value
                                best_gt_id = gt_id

                        if best_gt_id >= 0 and best_overlap > min_iou:
                            visited_gt[best_gt_id] = True

                            matches.append(predicted_id)
                            if len(matches) >= len(gt_bboxes):
                                break

                    tp = np.zeros([len(predicted_bboxes)], dtype=np.int32)
                    tp[matches] = 1

                    total_num_candidates += gt_bboxes.shape[0]
                    total_num_matches += len(matches)

                    if label not in out_matches:
                        out_matches[label] = {'tp': tp,
                                              'scores': predicted_scores,
                                              'num_images': 1,
                                              'num_gt': len(gt_bboxes)}
                    else:
                        last_data = out_matches[label]
                        out_matches[label] = {'tp': np.append(last_data['tp'], tp),
                                              'scores': np.append(last_data['scores'], predicted_scores),
                                              'num_images': last_data['num_images'] + 1,
                                              'num_gt': last_data['num_gt'] + len(gt_bboxes)}
                else:
                    tp = np.zeros([len(sample_predictions[label].scores)], dtype=np.int32)

                    if label not in out_matches:
                        out_matches[label] = {'tp': tp,
                                              'scores': sample_predictions[label].scores,
                                              'num_images': 1,
                                              'num_gt': 0}
                    else:
                        last_data = out_matches[label]
                        out_matches[label] = {'tp': np.append(last_data['tp'], tp),
                                              'scores': np.append(last_data['scores'],
                                                                  sample_predictions[label].scores),
                                              'num_images': last_data['num_images'] + 1,
                                              'num_gt': last_data['num_gt']}

        matched_ratio = float(total_num_matches) / float(total_num_candidates) if total_num_candidates > 0 else 0.0
        print('Matched GT bbox: {} / {} ({:.3f}%)'
              .format(total_num_matches, total_num_candidates, 1e2 * matched_ratio))

        return out_matches

    def _get_metrics(tp, scores, num_images, num_gt):
        def _ap(in_recall, in_precision):
            mrec = np.concatenate(([0.], in_recall, [1.]))
            mpre = np.concatenate(([0.], in_precision, [0.]))

            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            i = np.where(mrec[1:] != mrec[:-1])[0]

            return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        def _miss_rate(miss_rates, fppis):
            position = bisect(fppis, fppi_level)
            p1 = position - 1
            p2 = position if position < len(miss_rates) else p1
            return 0.5 * (miss_rates[p1] + miss_rates[p2])

        sorted_ind = np.argsort(-scores)

        tp_sorted = np.copy(tp[sorted_ind])
        fp_sorted = np.logical_not(tp)[sorted_ind]

        tp_cumsum = np.cumsum(tp_sorted)
        fp_cumsum = np.cumsum(fp_sorted)

        ind = len(scores) - np.unique(scores[sorted_ind[::-1]], return_index=True)[1] - 1
        ind = ind[::-1]

        tp_cumsum = tp_cumsum[ind]
        fp_cumsum = fp_cumsum[ind]

        recall = tp_cumsum / float(num_gt)
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
        miss_rates_values = 1.0 - recall
        fppis_values = fp_cumsum / float(num_images)

        # noinspection PyTypeChecker
        mr = _miss_rate(miss_rates_values, fppis_values)
        ap = _ap(recall, precision)

        return ap, mr

    all_matches = _match()

    ap_values = []
    mr_values = []
    metrics_by_class = {}
    for class_id, m in all_matches.iteritems():
        if len(m['tp']) == 0 or np.sum(m['tp']) == 0:
            ap_value, mr_value = 0.0, 1.0
        else:
            print('Debug. Num scores: {}, Num TP: {}, Num GT: {}, Num images: {}'
                  .format(len(m['scores']), len(m['tp']), m['num_gt'], m['num_images']))

            ap_value, mr_value = _get_metrics(m['tp'], m['scores'], m['num_images'], m['num_gt'])
        ap_values.append(ap_value)
        mr_values.append(mr_value)

        metrics_by_class[class_id] = ap_value, mr_value

    if return_all:
        return metrics_by_class
    else:
        map_metric = np.mean(ap_values) if len(ap_values) > 0 else 0.0
        mean_mr_metric = np.mean(mr_values) if len(mr_values) > 0 else 1.0
        return map_metric, mean_mr_metric


def calc_action_accuracy(predictions, gt, bg_class, num_classes, min_iou=0.5):
    """Calculates action normalized accuracy.

    :param predictions: Predicted boxes
    :param gt: Ground truth boxes
    :param bg_class: ID of background class
    :param num_classes: Number of classes
    :param min_iou: Min IoU value to match GT box
    :return: Accuracy scalar value
    """

    det_class = (bg_class + 1) % 2

    def _match(predicted_data, gt_data):
        total_num_candidates = 0
        total_num_matches = 0

        out_matches = {}
        for sample_id in trange(len(predicted_data), desc='Matching'):
            sample_predictions = predicted_data[sample_id][det_class]
            sample_gt = gt_data[sample_id][det_class]

            predicted_bboxes = sample_predictions.loc
            predicted_scores = sample_predictions.scores
            gt_bboxes = sample_gt.loc

            sorted_ind = np.argsort(-predicted_scores)
            predicted_bboxes = predicted_bboxes[sorted_ind]
            predicted_scores = predicted_scores[sorted_ind]
            predicted_original_ids = np.arange(len(predicted_scores))[sorted_ind]

            similarity_matrix = matrix_iou(predicted_bboxes, gt_bboxes)

            matches = []
            visited_gt = np.zeros(gt_bboxes.shape[0], dtype=np.bool)
            for predicted_id in xrange(predicted_bboxes.shape[0]):
                best_overlap = 0.0
                best_gt_id = -1
                for gt_id in xrange(gt_bboxes.shape[0]):
                    if visited_gt[gt_id]:
                        continue

                    overlap_value = similarity_matrix[predicted_id, gt_id]
                    if overlap_value > best_overlap:
                        best_overlap = overlap_value
                        best_gt_id = gt_id

                if best_gt_id >= 0 and best_overlap > min_iou:
                    visited_gt[best_gt_id] = True

                    matches.append((best_gt_id, predicted_original_ids[predicted_id]))
                    if len(matches) >= len(gt_bboxes):
                        break

            out_matches[sample_id] = {det_class: matches}

            total_num_candidates += gt_bboxes.shape[0]
            total_num_matches += len(matches)

        matched_ratio = float(total_num_matches) / float(total_num_candidates) if total_num_candidates > 0 else 0.0
        print('Matched GT bbox: {} / {} ({:.3f}%)'
              .format(total_num_matches, total_num_candidates, 1e2 * matched_ratio))

        return out_matches

    def _confusion_matrix(all_matched_ids, predicted_data, gt_data):
        out_cm = np.zeros([num_classes, num_classes], dtype=np.int32)
        for sample_id in all_matched_ids:
            sample_matched_ids = all_matched_ids[sample_id][det_class]
            sample_gt_labels = gt_data[sample_id][det_class].action_labels
            sample_pred_labels = predicted_data[sample_id][det_class].action_labels

            for gt_id, pred_id in sample_matched_ids:
                gt_label = sample_gt_labels[gt_id]
                pred_label = sample_pred_labels[pred_id]

                if gt_label >= num_classes:
                    continue

                out_cm[gt_label, pred_label] += 1

        return out_cm

    all_matches = _match(predictions, gt)
    cm = _confusion_matrix(all_matches, predictions, gt)

    return cm
