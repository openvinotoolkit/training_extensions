"""
 Copyright (c) 2020 Intel Corporation

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

# pylint: disable=too-many-branches, too-many-statements

import json
from bisect import bisect
from collections import namedtuple

import cv2
import mmcv
import numpy as np
from tqdm import tqdm

from mmdet import datasets # pylint: disable=import-error


def voc_ap(recall, precision, use_07_metric=False):
    """ average_precision = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        average_precision = 0.0
        for threshold in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= threshold) == 0:
                precision_at_threshold = 0
            else:
                precision_at_threshold = np.max(precision[recall >= threshold])
            average_precision += precision_at_threshold / 11.
    else:
        # Correct AP calculation.
        # First append sentinel values at the end.
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # Compute the precision envelope.
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # To calculate area under PR curve, look for points
        # where X axis (recall) changes value.
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # And sum (\Delta recall) * prec.
        average_precision = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return average_precision


def compute_miss_rate(miss_rates, fppis, fppi_level=0.1):
    """ Compute miss rate at fppi level. """

    position = bisect(fppis, fppi_level)
    position1 = position - 1
    position2 = position if position < len(miss_rates) else position1
    return 0.5 * (miss_rates[position1] + miss_rates[position2])


def evaluate_detections(ground_truth, predictions, class_name, overlap_threshold=0.5,
                        allow_multiple_matches_per_ignored=True,
                        verbose=True):
    """ Compute set of object detection quality metrics. """

    Detection = namedtuple('Detection', ['image', 'bbox', 'score', 'gt_match'])
    GT = namedtuple('GroundTruth', ['bbox', 'is_matched', 'is_ignored'])
    detections = [Detection(image=img_pred.image_path,
                            bbox=np.array(obj_pred["bbox"]),
                            score=obj_pred.get("score", 0.0),
                            gt_match=-1)
                  for img_pred in predictions
                  for obj_pred in img_pred
                  if obj_pred["type"] == class_name]

    scores = np.array([detection.score for detection in detections])
    sorted_ind = np.argsort(-scores)
    detections = [detections[i] for i in sorted_ind]

    gts = {}
    for img_gt in ground_truth:
        gts[img_gt.image_path] = GT(
            bbox=np.vstack([np.array(obj_gt["bbox"]) for obj_gt in img_gt]) if img_gt else np.empty(
                (0, 4)),
            is_matched=np.zeros(len(img_gt), dtype=bool),
            is_ignored=np.array([obj_gt.get("is_ignored", False) for obj_gt in img_gt], dtype=bool))

    detections_num = len(detections)
    true_pos = np.zeros(detections_num)
    false_pos = np.zeros(detections_num)

    for i, detection in tqdm(enumerate(detections), desc="Processing detections",
                             disable=not verbose):
        image_path = detection.image
        bboxes_gt = gts[image_path].bbox
        bbox = detection.bbox
        max_overlap = -np.inf

        if bboxes_gt is not None and bboxes_gt.shape[0] > 0:
            intersection_xmin = np.maximum(bboxes_gt[:, 0], bbox[0])
            intersection_ymin = np.maximum(bboxes_gt[:, 1], bbox[1])
            intersection_xmax = np.minimum(bboxes_gt[:, 0] + bboxes_gt[:, 2], bbox[0] + bbox[2])
            intersection_ymax = np.minimum(bboxes_gt[:, 1] + bboxes_gt[:, 3], bbox[1] + bbox[3])
            intersection_width = np.maximum(intersection_xmax - intersection_xmin, 0.)
            intersection_height = np.maximum(intersection_ymax - intersection_ymin, 0.)
            intersection = intersection_width * intersection_height

            det_area = bbox[2] * bbox[3]
            gt_area = bboxes_gt[:, 2] * bboxes_gt[:, 3]
            union = (det_area + gt_area - intersection)
            ignored_mask = gts[image_path].is_ignored
            if allow_multiple_matches_per_ignored:
                if np.any(ignored_mask):
                    union[ignored_mask] = det_area

            overlaps = intersection / union
            # Match not ignored ground truths first.
            if np.any(~ignored_mask):
                overlaps_filtered = np.copy(overlaps)
                overlaps_filtered[ignored_mask] = 0.0
                max_overlap = np.max(overlaps_filtered)
                argmax_overlap = np.argmax(overlaps_filtered)
            # If match with non-ignored ground truth is not good enough,
            # try to match with ignored ones.
            if max_overlap < overlap_threshold and np.any(ignored_mask):
                overlaps_filtered = np.copy(overlaps)
                overlaps_filtered[~ignored_mask] = 0.0
                max_overlap = np.max(overlaps_filtered)
                argmax_overlap = np.argmax(overlaps_filtered)
            detections[i] = detection._replace(gt_match=argmax_overlap)

        if max_overlap >= overlap_threshold:
            if not gts[image_path].is_ignored[argmax_overlap]:
                if not gts[image_path].is_matched[argmax_overlap]:
                    true_pos[i] = 1.
                    gts[image_path].is_matched[argmax_overlap] = True
                else:
                    false_pos[i] = 1.
            elif not allow_multiple_matches_per_ignored:
                gts[image_path].is_matched[argmax_overlap] = True
        else:
            false_pos[i] = 1.

    false_pos = np.cumsum(false_pos)
    true_pos = np.cumsum(true_pos)

    debug_visualization = False
    if debug_visualization:
        for image_path, bboxes_gt in gts.items():

            print(image_path)
            image = cv2.imread(image_path)
            image_gt = np.copy(image)
            for bbox in bboxes_gt.bbox:
                cv2.rectangle(image_gt, tuple(bbox[:2]), tuple(bbox[2:] + bbox[:2]),
                              color=(255, 255, 0), thickness=2)
            cv2.imshow("gt", image_gt)
            for detection in detections:
                if detection.image != image_path:
                    continue
                bbox = detection.bbox
                cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:] + bbox[:2]), color=(0, 255, 0),
                              thickness=2)
                if detection.gt_match is not None:
                    bbox = bboxes_gt.bbox[detection.gt_match]
                    cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:] + bbox[:2]),
                                  color=(0, 0, 255), thickness=1)
                cv2.imshow("image", image)
                cv2.waitKey(0)

    # Handle equal-score detections.
    # Get index of the last occurrence of a score.
    ind = len(scores) - np.unique(scores[sorted_ind[::-1]], return_index=True)[1] - 1
    ind = ind[::-1]
    # Though away redundant points.
    false_pos = false_pos[ind]
    true_pos = true_pos[ind]

    total_positives_num = np.sum([np.count_nonzero(~gt.is_ignored) for gt in gts.values()])
    recall = true_pos / float(total_positives_num)
    # Avoid divide by zero in case the first detection matches an ignored ground truth.
    precision = true_pos / np.maximum(true_pos + false_pos, np.finfo(np.float64).eps)
    miss_rate = 1.0 - recall
    fppi = false_pos / float(len(gts))

    return recall, precision, miss_rate, fppi


class ImageAnnotation:
    """ Represent image annotation. """

    def __init__(self, image_path, objects=None, ignore_regs=None):
        self.image_path = image_path
        self.objects = objects if objects else []
        self.ignore_regs = ignore_regs if ignore_regs else []

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, item):
        return self.objects[item]


def points_2_xywh(box):
    """ Converts [xmin, ymin, xmax, ymax] to [xmin, ymin, width, height]. """

    box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    box = [int(round(x)) for x in box]
    return box


def clip_bbox(bbox, im_size):
    """ Clips box. """

    bbox = np.maximum(np.copy(bbox), 0)
    xmin, ymin, width, height = bbox
    width = min(xmin + width, im_size[0]) - xmin
    height = min(ymin + height, im_size[1]) - ymin
    if width == 0 and height == 0:
        xmin = ymin = width = height = -1
    return np.array([xmin, ymin, width, height])


def voc_eval(result_file, dataset, iou_thr, image_size):
    """ VOC AP evaluation procedure for range of face sizes. """

    det_results = mmcv.load(result_file)
    min_detection_confidence = 0.01

    out = []

    for obj_size in ((10, 1024), (32, 1024), (64, 1024), (100, 1024)):

        groundtruth = []
        predictions = []

        for i, _ in enumerate(tqdm(dataset)):
            ann = dataset.get_ann_info(i)
            bboxes = ann['bboxes']

            # +1 is to compensate pre-processing in XMLDataset
            if isinstance(dataset, datasets.XMLDataset):
                bboxes = [np.array(bbox) + np.array((1, 1, 1, 1)) for bbox in bboxes]
            elif isinstance(dataset, datasets.CocoDataset):
                bboxes = [np.array(bbox) + np.array((0, 0, 1, 1)) for bbox in bboxes]
            # convert from [xmin, ymin, xmax, ymax] to [xmin, ymin, w, h]
            bboxes = [points_2_xywh(bbox) for bbox in bboxes]
            # clip bboxes
            bboxes = [clip_bbox(bbox, image_size) for bbox in bboxes]
            # filter out boxes with to small height or with invalid size (-1)
            ignored = [not (obj_size[0] <= b[3] <= obj_size[1]) or np.any(b == -1) for b in bboxes]
            objects = [{'bbox': bbox, 'is_ignored': ignore} for bbox, ignore in zip(bboxes, ignored)]
            groundtruth.append(ImageAnnotation(dataset.data_infos[i]['id'], objects))

            # filter out predictions with too low confidence
            detections = [{'bbox': points_2_xywh(bbox[:4]), 'score': bbox[4], 'type': 'face'} for
                          bbox
                          in det_results[i][0] if bbox[4] > min_detection_confidence]
            predictions.append(ImageAnnotation(dataset.data_infos[i]['id'], detections))

        recall, precision, miss_rates, fppis = evaluate_detections(
            groundtruth, predictions, 'face',
            allow_multiple_matches_per_ignored=True,
            overlap_threshold=iou_thr)

        miss_rate = compute_miss_rate(miss_rates, fppis) * 100
        average_precision = voc_ap(recall, precision) * 100

        print(f'image_size = {image_size}, '
              f'object_size = {obj_size}, '
              f'average_precision = {average_precision:.2f}%, '
              f'miss_rate = {miss_rate:.2f}%')

        average_precision = average_precision if not np.isnan(average_precision) else -1.0

        out.append({'image_size': image_size,
                    'object_size': obj_size,
                    'average_precision': average_precision,
                    'miss_rate': miss_rate})
    return out


def custom_voc_ap_evaluation(config, result_file, iou_thr, imsize, out, update_config):
    """ Main function. """

    cfg = mmcv.Config.fromfile(config)
    if update_config:
        cfg.merge_from_dict(update_config)
    if ',' in cfg.data.test.ann_file:
        cfg.data.test.ann_file = cfg.data.test.ann_file.split(',')
        cfg.data.test.img_prefix = cfg.data.test.img_prefix.split(',')
        assert len(cfg.data.test.ann_file) == len(cfg.data.test.img_prefix)

    test_dataset = datasets.builder.build_dataset(cfg.data.test)
    output = voc_eval(result_file, test_dataset, iou_thr, imsize)

    if out:
        with open(out, 'w') as write_file:
            json.dump(output, write_file, indent=4)
