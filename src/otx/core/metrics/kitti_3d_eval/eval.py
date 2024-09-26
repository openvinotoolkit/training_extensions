# flake8: noqa
# mypy: ignore-errors

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""KITTI 3D eval for OTX."""

from __future__ import annotations

import io as sysio
from typing import Any

import numba
import numpy as np
import torch

if torch.cuda.is_available():
    from .rotate_gpu_iou import rotate_iou_eval_gpu as rotate_iou_eval
else:
    from .rotate_iou import rotate_iou_eval_cpu as rotate_iou_eval


@numba.jit(nopython=True)
def get_thresholds(
    scores: np.ndarray,  # 1D array of confidence scores
    num_gt: int,  # Number of ground truth objects
    num_sample_pts: int = 41,  # Number of sample points used to compute recall thresholds
) -> np.ndarray:  # 1D array of recall thresholds
    """Compute recall thresholds for a given score array.

    Args:
        scores (np.ndarray): 1D array of confidence scores.
        num_gt (int): Number of ground truth objects.
        num_sample_pts (int, optional): Number of sample points used to
            compute recall thresholds. Defaults to 41.

    Returns:
        np.ndarray: 1D array of recall thresholds.
    """
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if ((r_recall - current_recall) < (current_recall - l_recall)) and (i < (len(scores) - 1)):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def clean_data(
    gt_anno: dict,  # ground truth annotations
    dt_anno: dict,  # detection results
    current_class: str,  # the current class name
    difficulty: int,  # the difficulty level
) -> tuple:  # (num_valid_gt, ignored_gt, ignored_dt, dc_bboxes)
    """Filter out the objects that are not in the current class.

    Args:
        gt_anno (dict): Ground truth annotations.
        dt_anno (dict): Detection results.
        current_class (str): The current class name.
        difficulty (int): The difficulty level.

    Returns:
        tuple: The number of valid objects, ignored_gt, ignored_dt, and dc_bboxes.
    """
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if gt_name == current_class:
            valid_class = 1
        elif current_class == "Pedestrian".lower() and "Person_sitting".lower() == gt_name:
            valid_class = 0
        elif current_class == "Car".lower() and "Van".lower() == gt_name:
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if (
            (gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
            or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
            or (height <= MIN_HEIGHT[difficulty])
        ):
            # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (ignore and (valid_class == 1)):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        # for i in range(num_gt):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if dt_anno["name"][i].lower() == current_class:
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(
    boxes: np.ndarray,  # shape: (N, 4)
    query_boxes: np.ndarray,  # shape: (K, 4)
    criterion: int = -1,  # default overlap criterion, -1: intersection over union, 0: intersection over box area, 1: intersection over query box area
) -> np.ndarray:  # shape: (N, K)
    """Args:
        boxes (np.ndarray): shape: (N, 4), 2D boxes, (x1, y1, x2, y2)
        query_boxes (np.ndarray): shape: (K, 4), 2D boxes, (x1, y1, x2, y2)
        criterion (int, optional): overlap criterion, -1: intersection over union, 0: intersection over box area, 1: intersection over query box area. Defaults to -1.

    Returns:
        np.ndarray: shape: (N, K), overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = (query_boxes[k, 2] - query_boxes[k, 0]) * (query_boxes[k, 3] - query_boxes[k, 1])
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0])
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1])
                if ih > 0:
                    if criterion == -1:
                        ua = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih
                    elif criterion == 0:
                        ua = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1])
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


@numba.jit(nopython=True)
def d3_box_overlap_kernel(
    boxes: np.ndarray,  # shape: (N, 7)
    qboxes: np.ndarray,  # shape: (K, 7)
    rinc: np.ndarray,  # shape: (N, K)
    criterion: int = -1,  # default overlap criterion
) -> None:
    """Args:
        boxes: Array of shape (N, 7) representing N 3D boxes.
        qboxes: Array of shape (K, 7) representing K 3D boxes.
        rinc: Array of shape (N, K) representing the overlap between boxes
            and qboxes.
        criterion: Overlap criterion. Defaults to -1. If -1, uses the
            intersection-over-union (IoU) criterion. If 0, uses the
            intersection-over-area1 criterion. If 1, uses the
            intersection-over-area2 criterion.

    Returns:
        None
    """
    # ONLY support overlap in CAMERA, not lidar.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = min(boxes[i, 1], qboxes[j, 1]) - max(boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4])

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = area1 + area2 - inc
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


@numba.jit(nopython=True)
def compute_statistics_jit(
    overlaps: np.ndarray,  # shape: (total_dt_num, total_gt_num)
    gt_datas: np.ndarray,  # shape: (total_gt_num, 7)
    dt_datas: np.ndarray,  # shape: (total_dt_num, 7)
    ignored_gt: list[int],  # shape: (total_gt_num)
    ignored_det: list[int],  # shape: (total_dt_num)
    dc_bboxes: np.ndarray,  # shape: (total_dc_num, 4)
    metric: int,
    min_overlap: float,
    thresh: float = 0,
    compute_fp: bool = False,
    compute_aos: bool = False,
) -> tuple[int, int, int, float, np.ndarray]:
    """This function computes statistics of an evaluation.

    Args:
        overlaps (np.ndarray): Overlap between dt and gt bboxes.
        gt_datas (np.ndarray): Ground truth data.
        dt_datas (np.ndarray): Detection data.
        ignored_gt (List[int]): Ignore ground truth indices.
        ignored_det (List[int]): Ignore detection indices.
        dc_bboxes (np.ndarray): Don't care bboxes.
        metric (int): Evaluation metric.
        min_overlap (float): Minimum overlap between dt and gt bboxes.
        thresh (float): Detection score threshold. Defaults to 0.
        compute_fp (bool): Whether to compute false positives. Defaults to False.
        compute_aos (bool): Whether to compute average orientation similarity. Defaults to False.

    Returns:
        Tuple[int, int, int, float, np.ndarray]: tp, fp, fn, similarity, thresholds
    """
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size,))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if ignored_det[j] == -1:
                continue
            if assigned_detection[j]:
                continue
            if ignored_threshold[j]:
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if not compute_fp and (overlap > min_overlap) and dt_score > valid_detection:
                det_idx = j
                valid_detection = dt_score
            elif (
                compute_fp
                and (overlap > min_overlap)
                and (overlap > max_overlap or assigned_ignored_det)
                and ignored_det[j] == 0
            ):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif compute_fp and (overlap > min_overlap) and (valid_detection == NO_DETECTION) and ignored_det[j] == 1:
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif (valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if not (assigned_detection[i] or ignored_det[i] == -1 or ignored_det[i] == 1 or ignored_threshold[i]):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if assigned_detection[j]:
                        continue
                    if ignored_det[j] == -1 or ignored_det[j] == 1:
                        continue
                    if ignored_threshold[j]:
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx,))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


@numba.jit(nopython=True)
def get_split_parts(num: int, num_part: int) -> list[int]:
    """Split a number into parts.

    Args:
        num (int): The number to split.
        num_part (int): The number of parts to split into.

    Returns:
        List[int]: A list of the parts.
    """
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(
    overlaps: np.ndarray,  # shape: (total_dt_num, total_gt_num)
    pr: np.ndarray,  # shape: (num_thresholds, 4)
    gt_nums: np.ndarray,  # shape: (num_samples)
    dt_nums: np.ndarray,  # shape: (num_samples)
    dc_nums: np.ndarray,  # shape: (num_samples)
    gt_datas: np.ndarray,  # shape: (total_gt_num, 7)
    dt_datas: np.ndarray,  # shape: (total_dt_num, 7)
    dontcares: np.ndarray,  # shape: (total_dc_num, 4)
    ignored_gts: np.ndarray,  # shape: (total_gt_num)
    ignored_dets: np.ndarray,  # shape: (total_dt_num)
    metric: int,
    min_overlap: float,
    thresholds: np.ndarray,  # shape: (num_thresholds)
    compute_aos: bool = False,
) -> None:
    """Fast compute statistics. Must be used in CAMERA coordinate system.

    Args:
    overlaps: 2D array of shape (total_dt_num, total_gt_num)
    [dt_num, gt_num] is the overlap between dt_num-th detection
    and gt_num-th ground truth
    pr: 2D array of shape (num_thresholds, 4)
    [t, 0] is the number of true positives at threshold t
    [t, 1] is the number of false positives at threshold t
    [t, 2] is the number of false negatives at threshold t
    [t, 3] is the similarity at threshold t
    gt_nums: 1D array of shape (num_samples)
    gt_nums[i] is the number of ground truths in i-th sample
    dt_nums: 1D array of shape (num_samples)
    dt_nums[i] is the number of detections in i-th sample
    dc_nums: 1D array of shape (num_samples)
    dc_nums[i] is the number of dontcare areas in i-th sample
    gt_datas: 2D array of shape (total_gt_num, 7)
    gt_datas[i] is the i-th ground truth box
    dt_datas: 2D array of shape (total_dt_num, 7)
    dt_datas[i] is the i-th detection box
    dontcares: 2D array of shape (total_dc_num, 4)
    dontcares[i] is the i-th dontcare area
    ignored_gts: 1D array of shape (total_gt_num)
    ignored_gts[i] is 1 if the i-th ground truth is ignored, 0 otherwise
    ignored_dets: 1D array of shape (total_dt_num)
    ignored_dets[i] is 1 if the i-th detection is ignored, 0 otherwise
    metric: Eval type. 0: bbox, 1: bev, 2: 3d
    min_overlap: Min overlap
    thresholds: 1D array of shape (num_thresholds)
    thresholds[i] is the i-th threshold
    compute_aos: Whether to compute aos
    """
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num : dt_num + dt_nums[i], gt_num : gt_num + gt_nums[i]]
            gt_data = gt_datas[gt_num : gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num : dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num : gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num : dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num : dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos,
            )
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(
    gt_annos: list[dict[str, Any]],
    dt_annos: list[dict[str, Any]],
    metric: int,
    num_parts: int = 50,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Fast iou algorithm. This function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.

    Args:
        gt_annos: List of dict, must from get_label_annos() in kitti_common.py
        dt_annos: List of dict, must from get_label_annos() in kitti_common.py
        metric: Eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: Int, a parameter for fast calculate algorithm

    Returns:
        Tuple of
            overlaps: List of numpy arrays, shape (num_gt, num_dt)
            parted_overlaps: List of numpy arrays, shape (num_gt, num_dt)
            total_gt_num: Numpy array, shape (num_images,)
            total_dt_num: Numpy array, shape (num_images,)
    """

    def d3_box_overlap(boxes, qboxes, criterion=-1):
        rinc = rotate_iou_eval(boxes[:, [0, 2, 3, 5, 6]], qboxes[:, [0, 2, 3, 5, 6]], 2)
        d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
        return rinc

    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx : example_idx + num_part]
        dt_annos_part = dt_annos[example_idx : example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx : example_idx + num_part]
        dt_annos_part = dt_annos[example_idx : example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx : gt_num_idx + gt_box_num, dt_num_idx : dt_num_idx + dt_box_num],
            )
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(
    gt_annos: list[dict[str, Any]],
    dt_annos: list[dict[str, Any]],
    current_class: str,
    difficulty: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, int]:
    """Prepare data for evaluation.

    Args:
        gt_annos (List[Dict[str, Any]]): Ground truth annotations.
        dt_annos (List[Dict[str, Any]]): Detection annotations.
        current_class (str): Current class name.
        difficulty (int): Difficulty level.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, int]:
            gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares, total_dc_num, total_num_valid_gt
    """
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate([gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        dt_datas = np.concatenate(
            [
                dt_annos[i]["bbox"],
                dt_annos[i]["alpha"][..., np.newaxis],
                dt_annos[i]["score"][..., np.newaxis],
            ],
            1,
        )
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares, total_dc_num, total_num_valid_gt)


def eval_class(
    gt_annos: list[dict[str, Any]],
    dt_annos: list[dict[str, Any]],
    current_classes: list[str],
    difficultys: list[int],
    metric: int,
    min_overlaps: np.ndarray,
    compute_aos: bool = False,
    num_parts: int = 50,
) -> dict[str, np.ndarray]:
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.

    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of label names
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (
                gt_datas_list,
                dt_datas_list,
                ignored_gts,
                ignored_dets,
                dontcares,
                total_dc_num,
                total_num_valid_gt,
            ) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False,
                    )
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(gt_datas_list[idx : idx + num_part], 0)
                    dt_datas_part = np.concatenate(dt_datas_list[idx : idx + num_part], 0)
                    dc_datas_part = np.concatenate(dontcares[idx : idx + num_part], 0)
                    ignored_dets_part = np.concatenate(ignored_dets[idx : idx + num_part], 0)
                    ignored_gts_part = np.concatenate(ignored_gts[idx : idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx : idx + num_part],
                        total_dt_num[idx : idx + num_part],
                        total_dc_num[idx : idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos,
                    )
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }
    return ret_dict


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def do_eval_cut_version(
    gt_annos: list[dict[str, Any]],  # type hint
    dt_annos: list[dict[str, Any]],  # type hint
    current_classes: list[str],  # type hint
    min_overlaps: np.ndarray,  # type hint
    compute_aos: bool = False,  # type hint
) -> tuple[float, float]:  # type hint
    """Evaluates detections with COCO style AP.

    Args:
        gt_annos (List[dict]): Ground truth annotations.
        dt_annos (List[dict]): Detection results.
        current_classes (List[str]): Classes to evaluate.
        min_overlaps (np.ndarray): Overlap ranges.
        compute_aos (bool): Whether to compute aos.

    Returns:
        Tuple[float, float]: Bounding box and 3D bounding box AP.
    """

    def _get_mAP(prec: np.ndarray) -> np.ndarray:
        sums = 0
        for i in range(0, prec.shape[-1], 4):
            sums = sums + prec[..., i]
        return sums / 11 * 100

    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2]
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0, min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    # get 2D bbox mAP
    mAP_bbox = _get_mAP(ret["precision"])

    # get 3D bbox mAP
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2, min_overlaps)
    mAP_3d = _get_mAP(ret["precision"])

    return mAP_bbox, mAP_3d


def get_coco_eval_result(
    gt_annos: list[dict],
    dt_annos: list[dict],
    current_classes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates detections with COCO style AP.

    Args:
        gt_annos (List[dict]): Ground truth annotations.
        dt_annos (List[dict]): Detection results.
        current_classes (List[str]): Classes to evaluate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Bounding box and 3D bounding box AP.
    """

    def do_coco_style_eval(
        gt_annos: list[dict],
        dt_annos: list[dict],
        current_classes: list[str],
        overlap_ranges: np.ndarray,
        compute_aos: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates detections with COCO style AP.

        Args:
            gt_annos (List[dict]): Ground truth annotations.
            dt_annos (List[dict]): Detection results.
            current_classes (List[str]): Classes to evaluate.
            overlap_ranges (np.ndarray): Overlap ranges.
            compute_aos (bool): Whether to compute aos.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Bounding box and 3D bounding box AP.
        """
        min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])

        for i in range(overlap_ranges.shape[1]):
            for j in range(overlap_ranges.shape[2]):
                min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j][:2], 10)

        mAP_bbox, mAP_3d = do_eval_cut_version(gt_annos, dt_annos, current_classes, min_overlaps, compute_aos)

        return mAP_bbox.mean(-1), mAP_3d.mean(-1)

    iou_range = [0.5, 0.95, 10]
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]

    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        # IoU from 0.5 to 0.95
        overlap_ranges[:, :, i] = np.array(iou_range)[:, np.newaxis]
    result = ""
    # check whether alpha is valid
    compute_aos = False
    mAPbbox, mAP3d = do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos)

    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(iou_range)[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str(f"{curcls} " "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range))
        result += print_str(f"bbox AP:{mAPbbox[j, 0]:.2f}, {mAPbbox[j, 1]:.2f}, {mAPbbox[j, 2]:.2f}")
        result += print_str(f"3d   AP:{mAP3d[j, 0]:.2f}, {mAP3d[j, 1]:.2f}, {mAP3d[j, 2]:.2f}")

    print("\n COCO style evaluation results: \n", result)

    return mAPbbox, mAP3d
