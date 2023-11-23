"""Collection of Action detection evaluiation utils.."""

# Copyright (C) 2021 Intel Corporation
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

import time
from collections import defaultdict

import numpy as np
from mmaction.core.evaluation.ava_evaluation import (
    object_detection_evaluation as mm_det_eval,
)
from mmaction.core.evaluation.ava_evaluation import standard_fields
from mmaction.core.evaluation.ava_utils import print_time, read_exclusions

from otx.utils.logger import get_logger

logger = get_logger()


# pylint: disable=too-many-locals, too-many-branches
def det_eval(predictions, result_type, labels, video_infos, exclude_file, verbose=True, custom_classes=None):
    """Evaluation method for AVA Dataset."""

    assert result_type in ["mAP"]

    start = time.time()
    categories, class_whitelist = _read_labelmap(labels)
    if custom_classes is not None:
        custom_classes = custom_classes[1:]
        assert set(custom_classes).issubset(set(class_whitelist))
        class_whitelist = custom_classes
        categories = [cat for cat in categories if cat["id"] in custom_classes]

    # loading gt, do not need gt score
    gt_boxes, gt_labels = _load_gt(video_infos)
    if verbose:
        print_time("Reading detection results", start)

    if exclude_file is not None:
        with open(exclude_file, encoding="utf-8") as ex_file:
            excluded_keys = read_exclusions(ex_file)
    else:
        excluded_keys = []

    start = time.time()
    boxes, labels, scores = predictions
    if verbose:
        print_time("Reading detection results", start)

    # Evaluation for mAP
    pascal_evaluator = mm_det_eval.PascalDetectionEvaluator(categories)

    start = time.time()
    for image_key in gt_boxes:
        if verbose and image_key in excluded_keys:
            logger.info("Found excluded timestamp in detections: %s. It will be ignored.", image_key)
            continue
        pascal_evaluator.add_single_ground_truth_image_info(
            image_key,
            {
                standard_fields.InputDataFields.groundtruth_boxes: np.array(gt_boxes[image_key], dtype=float),
                standard_fields.InputDataFields.groundtruth_classes: np.array(gt_labels[image_key], dtype=int),
            },
        )
    if verbose:
        print_time("Convert groundtruth", start)

    start = time.time()
    for image_key in boxes:
        if verbose and image_key in excluded_keys:
            logger.info("Found excluded timestamp in detections: %s. It will be ignored.", image_key)
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key,
            {
                standard_fields.DetectionResultFields.detection_boxes: np.array(boxes[image_key], dtype=float),
                standard_fields.DetectionResultFields.detection_classes: np.array(labels[image_key], dtype=int),
                standard_fields.DetectionResultFields.detection_scores: np.array(scores[image_key], dtype=float),
            },
        )
    if verbose:
        print_time("convert detections", start)

    start = time.time()
    metrics = pascal_evaluator.evaluate()
    if verbose:
        print_time("run_evaluator", start)
    for display_name, value in metrics.items():
        print(f"{display_name}=\t{value}")
    return {display_name: value for display_name, value in metrics.items() if "ByCategory" not in display_name}


def _read_labelmap(labels):
    """Generate label map from LabelEntity."""
    labelmap = []
    class_ids = set()
    for label in labels:
        labelmap.append({"id": int(label.id), "name": str(label.name)})
        class_ids.add(int(label.id))
    return labelmap, class_ids


def _load_gt(video_infos):
    """Generate ground truth information from video_infos."""
    boxes = defaultdict(list)
    labels = defaultdict(list)
    for video_info in video_infos:
        img_key = video_info["img_key"]
        gt_bboxes = video_info["gt_bboxes"]
        gt_labels = video_info["gt_labels"]
        for gt_label, gt_bbox in zip(gt_labels, gt_bboxes):
            for idx, val in enumerate(gt_label):
                if val == 1:
                    boxes[img_key].append(gt_bbox)
                    labels[img_key].append(idx)
    return boxes, labels
