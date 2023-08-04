"""Evaluator of OTX Detection."""

# Copyright (C) 2023 Intel Corporation
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

import multiprocessing as mp
from typing import Dict, List, Tuple

import mmcv
import numpy as np
import pycocotools.mask as mask_util
from mmcv.utils import print_log
from mmdet.core import BitmapMasks, PolygonMasks, eval_map
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.core.evaluation.class_names import get_classes
from mmdet.core.evaluation.mean_ap import average_precision
from terminaltables import AsciiTable

from otx.api.entities.label import Domain


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


def sanitize_coordinates(bbox: np.ndarray, height: int, width: int, padding=1) -> np.ndarray:
    """Sanitize coordinates of bounding boxes so that they fit within the image.

    Args:
        bbox (np.ndarray): bounding boxes with shape (4, )
        height (int): image height
        width (int): image width
        padding (int, optional): padding added to each side of the bounding box. Defaults to 1.

    Returns:
        np.ndarray: sanitized bounding boxes with shape (4, )
    """
    x1, y1, x2, y2 = bbox.astype(np.int)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return np.array([x1, y1, x2, y2])


def mask_iou(det: Tuple[np.ndarray, BitmapMasks], gt_masks: PolygonMasks) -> np.ndarray:
    """Compute the intersection over union between the detected masks and ground truth masks.

    Args:
        det (Tuple[np.ndarray, BitmapMasks]): detected bboxes and masks
        gt_masks (PolygonMasks): ground truth masks

    Note:
        It first compute IoU between bounding boxes, then compute IoU between masks
        if IoU between bounding boxes is greater than 0.
        Detection mask is resized to detected bounding box size and
        padded to the same size as ground truth mask in order to compute IoU.

    Returns:
        np.ndarray: iou between detected masks and ground truth masks

    """
    det_bboxes, det_masks = det
    gt_bboxes = gt_masks.get_bboxes()
    img_h, img_w = gt_masks.height, gt_masks.width
    ious = bbox_overlaps(det_bboxes, gt_bboxes, mode="iou")
    if not ious.any():
        return ious
    for coord in np.argwhere(ious > 0):
        m, n = coord
        det_bbox, det_mask = sanitize_coordinates(det_bboxes[m], img_h, img_w), det_masks[m]
        gt_bbox, gt_mask = sanitize_coordinates(gt_bboxes[n], img_h, img_w), gt_masks[n]
        # add padding to det_mask and gt_mask so that they have the same size
        min_x1 = min(det_bbox[0], gt_bbox[0])
        min_y1 = min(det_bbox[1], gt_bbox[1])
        max_x2 = max(det_bbox[2], gt_bbox[2])
        max_y2 = max(det_bbox[3], gt_bbox[3])
        det_bbox_h, det_bbox_w = det_bbox[3] - det_bbox[1], det_bbox[2] - det_bbox[0]
        det_mask = det_mask.resize((det_bbox_h, det_bbox_w))
        det_mask = det_mask.expand(max_y2 - min_y1, max_x2 - min_x1, det_bbox[1] - min_y1, det_bbox[0] - min_x1)
        gt_mask = gt_mask.crop(gt_bbox)
        gt_mask = gt_mask.to_bitmap()
        gt_mask = gt_mask.expand(max_y2 - min_y1, max_x2 - min_x1, gt_bbox[1] - min_y1, gt_bbox[0] - min_x1)
        # compute iou between det_mask and gt_mask
        det_mask = det_mask.to_ndarray()
        gt_mask = gt_mask.to_ndarray()
        assert det_mask.shape == gt_mask.shape, f"det_mask.shape={det_mask.shape} != gt_mask.shape={gt_mask.shape}"
        ious[m, n] = np.sum(det_mask & gt_mask) / np.sum(det_mask | gt_mask)
    return ious


def tpfpmiou_func(  # pylint: disable=too-many-locals
    det: Tuple[np.ndarray, BitmapMasks], gt_masks: PolygonMasks, cls_scores, iou_thr=0.5
):
    """Compute tp, fp, miou for each image.

    Args:
        det (Tuple[np.ndarray, BitmapMasks]): detected bboxes and masks
        gt_masks (PolygonMasks): ground truth polygons
        cls_scores (np.ndarray): class scores
        iou_thr (float, optional): IoU threshold. Defaults to 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: tp, fp, miou for each image
    """
    num_dets = len(det[0])  # M
    num_gts = len(gt_masks)  # N

    tp = np.zeros(num_dets, dtype=np.float32)  # pylint: disable=invalid-name
    fp = np.zeros(num_dets, dtype=np.float32)  # pylint: disable=invalid-name
    gt_covered_iou = np.zeros(num_gts, dtype=np.float32)

    if len(gt_masks) == 0:
        fp[...] = 1
        return tp, fp, 0.0
    if num_dets == 0:
        return tp, fp, 0.0

    ious = mask_iou(det, gt_masks)  # (M, N)

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


class Evaluator:
    """OTX Evaluator for mAP and mIoU.

    Args:
            annotation (list(dict)): ground truth annotation
            domain (Domain): OTX algorithm domain
            classes (list): list of classes
            nproc (int, optional): number of processes. Defaults to 4.
    """

    def __init__(self, annotation: List[Dict], domain: Domain, classes: List[str], nproc=4):
        self.domain = domain
        self.classes = classes
        self.num_classes = len(classes)
        if domain != Domain.DETECTION:
            self.annotation = self.get_gt_instance_masks(annotation)
        else:
            self.annotation = annotation
        self.nproc = nproc

    def get_gt_instance_masks(self, annotation: List[Dict]):
        """Format ground truth instance mask annotation.

        Args:
            annotation (List[Dict]): per-image ground truth annotation

        Returns:
            cls_anno_list: per-class ground truth instance mask list
        """
        cls_anno_list: List[List] = [[] for _ in range(self.num_classes)]
        for class_id in range(self.num_classes):
            for ann in annotation:
                gt_inds = ann["labels"] == class_id
                polygon_masks = []
                if gt_inds.any():
                    gt_inds = np.where(gt_inds == 1)[0]
                    polygon_masks = ann["masks"][gt_inds]
                cls_anno_list[class_id].append(polygon_masks)
        return cls_anno_list

    def get_mask_det_results(self, det_results: List[Tuple], class_id: int) -> Tuple[List, List]:
        """Get mask detection results for a specific class.

        Args:
            det_results (list(tuple)): detection results including bboxes and masks
            class_id (int): class index

        Returns:
            cls_dets: per-class detection results including bboxes and decoded masks
            cls_scores: class scores
        """
        cls_scores = [img_res[0][class_id][..., -1] for img_res in det_results]
        cls_dets: List[Tuple] = []
        for det in det_results:
            det_bboxes = det[0][class_id][:, :4]
            det_masks = det[1][class_id]
            if len(det_masks) == 0:
                cls_dets.append(([], []))
            else:
                # Convert 28x28 encoded RLE mask detection to 28x28 BitmapMasks.
                det_masks = mask_util.decode(det_masks)
                det_masks = det_masks.transpose(2, 0, 1)
                det_masks = BitmapMasks(det_masks, *det_masks.shape[1:])
                cls_dets.append((det_bboxes, det_masks))
        return cls_dets, cls_scores

    def evaluate_mask(self, results, logger, iou_thr):
        """Evaluate mask results.

        Args:
            results (list): list of prediction
            logger (Logger): OTX logger
            iou_thr (float): IoU threshold

        Returns:
            metric: mAP and mIoU metric
        """
        assert len(results) == len(self.annotation[0]), "number of images should be equal!"
        num_imgs = len(results)
        eval_results = []

        ctx = mp.get_context("spawn")
        with ctx.Pool(self.nproc) as p:
            for class_id in range(self.num_classes):
                # get gt and det bboxes of this class
                cls_dets, cls_scores = self.get_mask_det_results(results, class_id)
                cls_gts = self.annotation[class_id]

                # compute tp and fp for each image with multiple processes
                tpfpmiou = p.starmap(
                    tpfpmiou_func, zip(cls_dets, cls_gts, cls_scores, [iou_thr for _ in range(num_imgs)])
                )
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
                ap = average_precision(recalls, precisions, "area")  # pylint: disable=invalid-name
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

        print_map_summary(mean_ap, eval_results, self.classes, None, logger=logger)

        return metrics["mAP"], eval_results

    def evaluate(self, results, logger, iou_thr, scale_ranges):
        """Evaluate detection results.

        Args:
            results (list): list of prediction
            logger (Logger): OTX logger
            iou_thr (float): IoU threshold
            scale_ranges (list): scale range for object detection evaluation

        Returns:
            metric: mAP and mIoU metric
        """
        if self.domain == Domain.DETECTION:
            return eval_map(
                results,
                self.annotation,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.classes,
                logger=logger,
            )
        return self.evaluate_mask(results, logger, iou_thr)
