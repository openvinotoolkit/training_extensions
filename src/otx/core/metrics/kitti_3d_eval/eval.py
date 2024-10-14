# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""KITTI 3D eval for OTX."""


from __future__ import annotations

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
    current_recall = 0.0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        r_recall = (i + 2) / num_gt if i < len(scores) - 1 else l_recall
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
    min_height = 20
    max_occlusion = 2
    max_truncation = 0.5
    ignored_gt, ignored_dt = [], []
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
        elif (current_class == "Pedestrian".lower() and "Person_sitting".lower() == gt_name) or (
            current_class == "Car".lower() and "Van".lower() == gt_name
        ):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if (
            (gt_anno["occluded"][i] > max_occlusion)
            or (gt_anno["truncated"][i] > max_truncation)
            or (height <= min_height)
        ):  # filter extrim cases
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (ignore and (valid_class == 1)):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

    for i in range(num_dt):
        valid_class = 1 if dt_anno["name"][i].lower() == current_class else -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < min_height:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt


@numba.jit(nopython=True)
def image_box_overlap(
    boxes: np.ndarray,  # shape: (N, 4)
    query_boxes: np.ndarray,  # shape: (K, 4)
    criterion: int = -1,  # default overlap criterion: intersection over union
) -> np.ndarray:  # shape: (N, K)
    """Image box overlap.

    Args:
        boxes (np.ndarray): shape: (N, 4), 2D boxes, (x1, y1, x2, y2)
        query_boxes (np.ndarray): shape: (K, 4), 2D boxes, (x1, y1, x2, y2)
        criterion (int, optional): overlap criterion, -1: intersection over union,
            0: intersection over box area, 1: intersection over query box area. Defaults to -1.

    Returns:
        np.ndarray: shape: (N, K), overlap between boxes and query_boxes
    """
    num_n = boxes.shape[0]
    num_k = query_boxes.shape[0]
    overlaps = np.zeros((num_n, num_k), dtype=boxes.dtype)
    for k in range(num_k):
        qbox_area = (query_boxes[k, 2] - query_boxes[k, 0]) * (query_boxes[k, 3] - query_boxes[k, 1])
        for n in range(num_n):
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
    boxes: np.ndarray,  # shape: (n, 7)
    qboxes: np.ndarray,  # shape: (k, 7)
    rinc: np.ndarray,  # shape: (n, k)
    criterion: int = -1,  # default overlap criterion
) -> None:
    """Calculate 3D box overlap.

    Args:
        boxes (np.ndarray): Array of shape (n, 7) representing n 3D boxes.
        qboxes (np.ndarray): Array of shape (k, 7) representing k 3D boxes.
        rinc (np.ndarray): Array of shape (n, k) representing the overlap between boxes
            and qboxes.
        criterion (int, optional): Overlap criterion. Defaults to -1. If -1, uses the
            intersection-over-union (IoU) criterion. If 0, uses the
            intersection-over-area1 criterion. If 1, uses the
            intersection-over-area2 criterion.

    Returns:
        None
    """
    # ONLY support overlap in CAMERA, not lidar.
    n, k = boxes.shape[0], qboxes.shape[0]
    for i in range(n):
        for j in range(k):
            if rinc[i, j] > 0:
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
    min_overlap: float,
    thresh: float = 0,
    compute_fp: bool = False,
) -> tuple[int, int, int, float, np.ndarray]:
    """This function computes statistics of an evaluation.

    Args:
        overlaps (np.ndarray): Overlap between dt and gt bboxes.
        gt_datas (np.ndarray): Ground truth data.
        dt_datas (np.ndarray): Detection data.
        ignored_gt (List[int]): Ignore ground truth indices.
        ignored_det (List[int]): Ignore detection indices.
        min_overlap (float): Minimum overlap between dt and gt bboxes.
        thresh (float): Detection score threshold. Defaults to 0.
        compute_fp (bool): Whether to compute false positives. Defaults to False.

    Returns:
        Tuple[int, int, int, float, np.ndarray]: tp, fp, fn, similarity, thresholds
    """
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True
    no_detection = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = no_detection
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
            elif compute_fp and (overlap > min_overlap) and (valid_detection == no_detection) and ignored_det[j] == 1:
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == no_detection) and ignored_gt[i] == 0:
            fn += 1
        elif (valid_detection != no_detection) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1):
            assigned_detection[det_idx] = True
        elif valid_detection != no_detection:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if not (assigned_detection[i] or ignored_det[i] == -1 or ignored_det[i] == 1 or ignored_threshold[i]):
                fp += 1
        nstuff = 0
        fp -= nstuff

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
    return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(
    overlaps: np.ndarray,  # shape: (total_dt_num, total_gt_num)
    pr: np.ndarray,  # shape: (num_thresholds, 4)
    gt_nums: np.ndarray,  # shape: (num_samples)
    dt_nums: np.ndarray,  # shape: (num_samples)
    gt_datas: np.ndarray,  # shape: (total_gt_num, 7)
    dt_datas: np.ndarray,  # shape: (total_dt_num, 7)
    ignored_gts: np.ndarray,  # shape: (total_gt_num)
    ignored_dets: np.ndarray,  # shape: (total_dt_num)
    min_overlap: float,
    thresholds: np.ndarray,  # shape: (num_thresholds)
) -> None:
    """Fast compute statistics. Must be used in CAMERA coordinate system.

    Args:
        overlaps (np.ndarray): 2D array of shape (total_dt_num, total_gt_num),
            [dt_num, gt_num] is the overlap between dt_num-th detection
            and gt_num-th ground truth
            pr (np.ndarray): 2D array of shape (num_thresholds, 4)
            [t, 0] is the number of true positives at threshold t
            [t, 1] is the number of false positives at threshold t
            [t, 2] is the number of false negatives at threshold t
            [t, 3] is the similarity at threshold t
        gt_nums (np.ndarray): 1D array of shape (num_samples),
            gt_nums[i] is the number of ground truths in i-th sample
        dt_nums (np.ndarray): 1D array of shape (num_samples),
            dt_nums[i] is the number of detections in i-th sample
        gt_datas (np.ndarray): 2D array of shape (total_gt_num, 7),
            gt_datas[i] is the i-th ground truth box
        dt_datas (np.ndarray): 2D array of shape (total_dt_num, 7),
            dt_datas[i] is the i-th detection box
        ignored_gts (np.ndarray): 1D array of shape (total_gt_num),
            ignored_gts[i] is 1 if the i-th ground truth is ignored, 0 otherwise
        ignored_dets (np.ndarray): 1D array of shape (total_dt_num),
            ignored_dets[i] is 1 if the i-th detection is ignored, 0 otherwise
        min_overlap (float): Min overlap
        thresholds (np.ndarray): 1D array of shape (num_thresholds),
            thresholds[i] is the i-th threshold
    """
    gt_num = 0
    dt_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num : dt_num + dt_nums[i], gt_num : gt_num + gt_nums[i]]
            gt_data = gt_datas[gt_num : gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num : dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num : gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num : dt_num + dt_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
            )
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]


def calculate_iou_partly(
    gt_annos: list[dict[str, Any]],
    dt_annos: list[dict[str, Any]],
    num_parts: int = 50,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Fast iou algorithm.

    This function can be used independently to do result analysis.
    Must be used in CAMERA coordinate system.

    Args:
        gt_annos: List of dict, must from get_label_annos() in kitti_common.py
        dt_annos: List of dict, must from get_label_annos() in kitti_common.py
        num_parts: Int, a parameter for fast calculate algorithm

    Returns:
        Tuple of
            overlaps: List of numpy arrays, shape (num_gt, num_dt)
            parted_overlaps: List of numpy arrays, shape (num_gt, num_dt)
            total_gt_num: Numpy array, shape (num_images,)
            total_dt_num: Numpy array, shape (num_images,)
    """

    def d3_box_overlap(boxes: np.ndarray, qboxes: np.ndarray, criterion: int = -1) -> np.ndarray:
        """Calculate 3D box overlap.

        Args:
            boxes (np.ndarray): Array of shape (n, 7) representing n 3D boxes.
            qboxes (np.ndarray): Array of shape (k, 7) representing k 3D boxes.
            criterion (int, optional): Overlap criterion. Defaults to -1. If -1, uses the
                intersection-over-union (IoU) criterion. If 0, uses the
                intersection-over-area1 criterion. If 1, uses the
                intersection-over-area2 criterion.

        Returns:
            np.ndarray: 1D array of shape (k, )
        """
        rinc = rotate_iou_eval(boxes[:, [0, 2, 3, 5, 6]], qboxes[:, [0, 2, 3, 5, 6]], 2)
        d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
        return rinc

    if len(gt_annos) != len(dt_annos):
        msg = "gt_annos and dt_annos must have same length"
        raise ValueError(msg)

    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx : example_idx + num_part]
        dt_annos_part = dt_annos[example_idx : example_idx + num_part]

        loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
        dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
        rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
        gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
        dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
        rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
        dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)

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
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, int]:
    """Prepare data for evaluation.

    Args:
        gt_annos (List[Dict[str, Any]]): Ground truth annotations.
        dt_annos (List[Dict[str, Any]]): Detection annotations.
        current_class (str): Current class name.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray],
        List[np.ndarray], List[np.ndarray], np.ndarray, int]:
            gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
            dontcares, total_num_valid_gt
    """
    gt_datas_list = []
    dt_datas_list = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class)
        num_valid_gt, ignored_gt, ignored_det = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))

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

    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, total_num_valid_gt)


def eval_class(
    gt_annos: list[dict[str, Any]],
    dt_annos: list[dict[str, Any]],
    current_classes: list[str],
    min_overlaps: np.ndarray,
    num_parts: int = 50,
) -> dict[str, np.ndarray]:
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.

    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of label names
        min_overlaps: float, min overlap. format: [num_overlap, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision
    """
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    num_samples_pts = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    precision = np.zeros([num_class, num_minoverlap, num_samples_pts])
    recall = np.zeros([num_class, num_minoverlap, num_samples_pts])
    for m, current_class in enumerate(current_classes):
        (
            gt_datas_list,
            dt_datas_list,
            ignored_gts,
            ignored_dets,
            total_num_valid_gt,
        ) = _prepare_data(gt_annos, dt_annos, current_class)
        for k, min_overlap in enumerate(min_overlaps[:, m]):
            thresholdss = []
            for i in range(len(gt_annos)):
                tp, fp, fn, similarity, thresholds = compute_statistics_jit(
                    overlaps[i],
                    gt_datas_list[i],
                    dt_datas_list[i],
                    ignored_gts[i],
                    ignored_dets[i],
                    min_overlap=min_overlap,
                    thresh=0.0,
                    compute_fp=False,
                )
                thresholdss += thresholds.tolist()
            thresholdss = np.array(thresholdss)
            thresholds = get_thresholds(thresholdss, total_num_valid_gt)
            thresholds = np.array(thresholds)
            pr = np.zeros([len(thresholds), 4])
            idx = 0
            for j, num_part in enumerate(split_parts):
                gt_datas_part = np.concatenate(gt_datas_list[idx : idx + num_part], 0)
                dt_datas_part = np.concatenate(dt_datas_list[idx : idx + num_part], 0)
                ignored_dets_part = np.concatenate(ignored_dets[idx : idx + num_part], 0)
                ignored_gts_part = np.concatenate(ignored_gts[idx : idx + num_part], 0)
                fused_compute_statistics(
                    parted_overlaps[j],
                    pr,
                    total_gt_num[idx : idx + num_part],
                    total_dt_num[idx : idx + num_part],
                    gt_datas_part,
                    dt_datas_part,
                    ignored_gts_part,
                    ignored_dets_part,
                    min_overlap=min_overlap,
                    thresholds=thresholds,
                )
                idx += num_part
            for i in range(len(thresholds)):
                recall[m, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                precision[m, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])

            for i in range(len(thresholds)):
                precision[m, k, i] = np.max(precision[m, k, i:], axis=-1)
                recall[m, k, i] = np.max(recall[m, k, i:], axis=-1)

    return {
        "recall": recall,
        "precision": precision,
    }


def do_eval_cut_version(
    gt_annos: list[dict[str, Any]],
    dt_annos: list[dict[str, Any]],
    current_classes: list[str],
    min_overlaps: np.ndarray,
) -> tuple[float, float]:
    """Evaluates detections with COCO style AP.

    Args:
        gt_annos (list[dict]): Ground truth annotations.
        dt_annos (list[dict]): Detection results.
        current_classes (list[str]): Classes to evaluate.
        min_overlaps (np.ndarray): Overlap ranges.

    Returns:
        Tuple[float, float]: Bounding box and 3D bounding box AP.
    """

    def _get_map(prec: np.ndarray) -> np.ndarray:
        sums = 0
        for i in range(0, prec.shape[-1], 4):
            sums = sums + prec[..., i]
        return sums / 11

    # min_overlaps: [num_minoverlap, num_class]
    # get 3D bbox mAP
    ret = eval_class(gt_annos, dt_annos, current_classes, min_overlaps)
    map_3d = _get_map(ret["precision"])

    return map_3d


def get_coco_eval_result(
    gt_annos: list[dict],
    dt_annos: list[dict],
    current_classes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates detections with COCO style AP.

    Args:
        gt_annos (list[dict]): Ground truth annotations.
        dt_annos (list[dict]): Detection results.
        current_classes (list[str]): Classes to evaluate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Bounding box and 3D bounding box AP.
    """

    def do_coco_style_eval(
        gt_annos: list[dict],
        dt_annos: list[dict],
        current_classes: list[str],
        overlap_ranges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates detections with COCO style AP.

        Args:
            gt_annos (list[dict]): Ground truth annotations.
            dt_annos (list[dict]): Detection results.
            current_classes (list[str]): Classes to evaluate.
            overlap_ranges (np.ndarray): Overlap ranges.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Bounding box and 3D bounding box AP.
        """
        min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])

        for i in range(overlap_ranges.shape[1]):
            min_overlaps[:, i] = np.linspace(*overlap_ranges[:, i], 10)

        map_3d = do_eval_cut_version(gt_annos, dt_annos, current_classes, min_overlaps)

        return map_3d.mean(-1)

    iou_range = [0.5, 0.95]
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]

    overlap_ranges = np.zeros([2, len(current_classes)])
    for i in range(len(current_classes)):
        # iou from 0.5 to 0.95
        overlap_ranges[:, i] = np.array(iou_range)

    return do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges)
