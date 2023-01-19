"""Evaluate mean AP for segmentation."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from multiprocessing import Pool
from typing import Dict, List

import mmcv
import numpy as np
import pycocotools.mask as mask_util
from mmcv.utils import print_log
from mmdet.core.evaluation.class_names import get_classes
from mmdet.core.evaluation.mean_ap import average_precision
from mmdet.core.mask.structures import PolygonMasks
from terminaltables import AsciiTable


def print_map_summary(  # pylint: disable=too-many-locals,too-many-branches
    mean_ap, results, dataset=None, scale_ranges=None, logger=None
):
    """Print mAP/mIoU and results of each class.

    A table will be printed to show the gts/dets/recall/AP/IoU of each class
    and the mAP/mIoU.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == "silent":
        return

    if isinstance(results[0]["ap"], np.ndarray):
        num_scales = len(results[0]["ap"])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    mious = np.zeros((num_scales, num_classes), dtype=np.float32)
    for i, cls_result in enumerate(results):
        if cls_result["recall"].size > 0:
            recalls[:, i] = np.array(cls_result["recall"], ndmin=2)[:, -1]
        aps[:, i] = cls_result["ap"]
        mious[:, i] = cls_result["miou"]
        num_gts[:, i] = cls_result["num_gts"]

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    elif mmcv.is_str(dataset):
        label_names = get_classes(dataset)
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ["class", "gts", "dets", "recall", "ap", "miou"]
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f"Scale range {scale_ranges[i]}", logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j],
                num_gts[i, j],
                results[j]["num_dets"],
                f"{recalls[i, j]:.3f}",
                f"{aps[i, j]:.3f}",
                f"{mious[i, j]:.3f}",
            ]
            table_data.append(row_data)
        table_data.append(["mAP", "", "", "", f"{mean_ap[i]:.3f}", f"{np.mean(mious[i]):.3f}"])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log("\n" + table.table, logger=logger)


def tpfpmiou_func(  # pylint: disable=too-many-locals
    det_masks: List[Dict],
    gt_masks: List[Dict],
    cls_scores,
    iou_thr=0.5,
):
    """Calculate Mean Intersection and Union (mIoU) and AP across predicted masks and GT masks.

    Args:
        det_masks: Detected masks of this image, with list size (m)
        gt_masks: GT masks of this image, with list size (n).
        cls_scores: (n, 1)
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.

    Returns:
        tuple[np.ndarray, np.ndarray, float]:
            <tp, fp> with elements of 0 and 1, the shape of each array is (m).
            <gt_covered_iou>: the average IoU between predicted mask and GT mask
    """
    num_dets = len(det_masks)
    num_gts = len(gt_masks)

    tp = np.zeros(num_dets, dtype=np.float32)  # pylint: disable=invalid-name
    fp = np.zeros(num_dets, dtype=np.float32)  # pylint: disable=invalid-name
    gt_covered_iou = np.zeros(num_gts, dtype=np.float32)

    if len(gt_masks) == 0:
        fp[...] = 1
        return tp, fp, 0.0
    if num_dets == 0:
        return tp, fp, 0.0

    ious = mask_util.iou(det_masks, gt_masks, len(gt_masks) * [0])
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-cls_scores)

    gt_covered = np.zeros(num_gts, dtype=bool)
    # if no area range is specified, gt_area_ignore is all False
    for i in sort_inds:
        if ious_max[i] >= iou_thr:
            matched_gt = ious_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                gt_covered_iou[matched_gt] = ious_max[i]
                tp[i] = 1
            else:
                fp[i] = 1
            # otherwise ignore this detected bbox, tp = 0, fp = 0
        else:
            fp[i] = 1
    return tp, fp, np.mean(gt_covered_iou)


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[tuple]): Same as `eval_segm()`.
        annotations (list[dict]): Same as `eval_segm()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[list[dict]], list[list[dict]], list[np.ndarray]]:
          <detected masks>: list[dict] includes all the predicted RLE masks
              for an image.
          <gt masks>: list[dict] includes all the GT RLE masks for an image
          <detected box scores>: each array in the list describes the predicted
              box scores for an image.
    """
    cls_scores = [img_res[0][class_id][..., -1] for img_res in det_results]
    cls_dets = []
    for i, det in enumerate(det_results):
        det_masks = det[1][class_id]
        cls_dets.append([])
        for det_mask in det_masks:
            if isinstance(det_mask, np.ndarray):
                cls_dets[i].append(mask_util.encode(np.array(det_mask[:, :, np.newaxis], order="F", dtype="uint8"))[0])
            else:
                cls_dets[i].append(det_mask)

    cls_gts = []
    for ann in annotations:
        gt_inds = ann["labels"] == class_id
        if isinstance(ann["masks"], PolygonMasks):
            masks = ann["masks"].to_ndarray()[gt_inds]
            encoded_masks = [
                mask_util.encode(np.array(m[:, :, np.newaxis], order="F", dtype="uint8"))[0] for m in masks
            ]
            cls_gts.append(encoded_masks)
        elif isinstance(ann["masks"], list):
            cls_gts.append([])
        else:
            raise RuntimeError("Unknown annotation format")

    return cls_dets, cls_gts, cls_scores


def eval_segm(  # pylint: disable=too-many-locals
    det_results,
    annotations,
    iou_thr=0.5,
    dataset=None,
    logger=None,
    nproc=4,
    metric="mAP",
):
    """Evaluate mAP/mIoU of a dataset.

    Args:
        det_results (list[tuple[list, list]]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner tuple indicates box
            and mask prediction. Each list of a predicted type (box/mask)
            includes the per-class detection of that type.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `masks`: numpy array of shape (k, 4)
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing tpfpmiou_func. Default: 4.

    Returns:
        tuple: (mIoU, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_classes = len(det_results[0][0])

    pool = Pool(nproc)  # pylint: disable=consider-using-with
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_results = get_cls_results(det_results, annotations, i)
        cls_dets, cls_gts, cls_scores = cls_results

        # compute tp and fp for each image with multiple processes
        tpfpmiou = pool.starmap(tpfpmiou_func, zip(cls_dets, cls_gts, cls_scores, [iou_thr for _ in range(num_imgs)]))
        tp, fp, miou = tuple(zip(*tpfpmiou))  # pylint: disable=invalid-name

        # sort all det bboxes by score, also sort tp and fp
        cls_scores = np.hstack(cls_scores)
        num_dets = cls_scores.shape[0]
        num_gts = np.sum([len(cls_gts) for cls_gts in cls_gts])
        sort_inds = np.argsort(cls_scores)[::-1]
        tp = np.hstack(tp)[sort_inds]  # pylint: disable=invalid-name
        fp = np.hstack(fp)[sort_inds]  # pylint: disable=invalid-name
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp)  # pylint: disable=invalid-name
        fp = np.cumsum(fp)  # pylint: disable=invalid-name
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)
        miou = np.mean(np.stack(miou))
        # calculate AP
        mode = "area" if dataset != "voc07" else "11points"
        ap = average_precision(recalls, precisions, mode)  # pylint: disable=invalid-name
        eval_results.append(
            {
                "num_gts": num_gts,
                "num_dets": num_dets,
                "recall": recalls,
                "precision": precisions,
                "ap": ap,
                "miou": miou,
            }
        )
    pool.close()

    metrics = {"mAP": 0.0, "mIoU": 0.0}
    mious, aps = [], []
    for cls_result in eval_results:
        if cls_result["num_gts"] > 0:
            aps.append(cls_result["ap"])
            mious.append(cls_result["miou"])
    mean_ap = np.array(aps).mean().item() if aps else 0.0
    mean_miou = np.array(mious).mean().item() if mious else 0.0
    metrics["mAP"] = mean_ap
    metrics["mIoU"] = mean_miou

    print_map_summary(mean_ap, eval_results, dataset, None, logger=logger)

    return metrics[metric], eval_results
