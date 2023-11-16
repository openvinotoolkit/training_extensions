"""Evaluator of OTX Detection."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import multiprocessing as mp
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import logging

import numpy as np
import pycocotools.mask as mask_util
import torch
from mmdet.evaluation.functional.class_names import get_classes
from mmdet.evaluation.functional.mean_ap import average_precision, eval_map
from mmdet.evaluation.metrics import VOCMetric
from mmdet.registry import METRICS
from mmdet.structures.bbox import bbox_overlaps
from mmdet.structures.mask import encode_mask_results
from mmdet.structures.mask.structures import BitmapMasks, PolygonMasks
from mmengine.logging import MMLogger, print_log
from mmengine.utils.misc import is_str
from terminaltables import AsciiTable

from otx.v2.api.entities.label import Domain


class ValueDifferenceError(Exception):
    """Custom Exception when two values are different when they should be same."""

    def __init__(self, value1: int | tuple[int], value2: int | tuple[int]) -> None:
        """Initialize Exception.

        Args:
            value1: (int | tuple[int])
            value2: (int | tuple[int])
        """
        super().__init__(f"{value1} and {value2} should be same.")


def print_map_summary(
    mean_ap: float | list,
    results: list[dict],
    dataset: list[str] | None = None,
    scale_ranges: list[tuple] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Print mAP/mIoU and results of each class.

    A table will be printed to show the gts/dets/recall/AP/IoU of each class
    and the mAP/mIoU.

    Args:
        mean_ap (float | list): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """
    if logger == "silent":
        return

    num_scales = len(results[0]["ap"]) if isinstance(results[0]["ap"], np.ndarray) else 1

    if scale_ranges is not None and len(scale_ranges) != num_scales:
        raise ValueDifferenceError(len(scale_ranges), num_scales)

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
    elif is_str(dataset):
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


def sanitize_coordinates(bbox: np.ndarray, height: int, width: int, padding: int | None = 1) -> np.ndarray:
    """Sanitize coordinates of bounding boxes so that they fit within the image.

    Args:
        bbox (np.ndarray): bounding boxes with shape (4, )
        height (int): image height
        width (int): image width
        padding (int, optional): padding added to each side of the bounding box. Defaults to 1.

    Returns:
        np.ndarray: sanitized bounding boxes with shape (4, )
    """
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return np.array([x1, y1, x2, y2])


def mask_iou(det: tuple[np.ndarray, BitmapMasks], gt_masks: PolygonMasks, iou_thr: float) -> np.ndarray:
    """Compute the intersection over union between the detected masks and ground truth masks.

    Args:
        det (Tuple[np.ndarray, BitmapMasks]): detected bboxes and masks
        gt_masks (PolygonMasks): ground truth masks
        iou_thr (float): IoU threshold

    Note:
        It first compute IoU between bounding boxes, then compute IoU between masks
        if IoU between bounding boxes is greater than 0.
        Detection mask is resized to detected bounding box size and
        padded to the same size as ground truth mask in order to compute IoU.

    Returns:
        np.ndarray: iou between detected masks and ground truth masks

    """
    det_bboxes, det_masks = det
    gt_bboxes = gt_masks.get_bboxes("hbox")
    img_h, img_w = gt_masks.height, gt_masks.width
    ious = bbox_overlaps(torch.from_numpy(det_bboxes), gt_bboxes.tensor, mode="iou").numpy()
    ious[ious < iou_thr] = 0.0
    if not ious.any():
        return ious
    # NOTE: further speed optimization (vectorization) could be done here
    for coord in np.argwhere(ious):
        x, y = coord
        det_bbox, det_mask = sanitize_coordinates(det_bboxes[x], img_h, img_w), det_masks[x]
        gt_bbox, gt_mask = sanitize_coordinates(gt_bboxes.numpy()[y], img_h, img_w), gt_masks[y]
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
        if det_mask.shape != gt_mask.shape:
            raise ValueDifferenceError(det_mask.shape, gt_mask.shape)
        ious[x, y] = np.sum(det_mask & gt_mask) / np.sum(det_mask | gt_mask)
    return ious


def compare_gts_and_predicitons(
    det: tuple[np.ndarray, (BitmapMasks | list)],
    gt_masks: PolygonMasks,
    cls_scores: np.ndarray,
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute tp, fp, miou for each image.

    Args:
        det (Tuple[np.ndarray, BitmapMasks]): detected bboxes and masks
        gt_masks (PolygonMasks): ground truth polygons
        cls_scores (np.ndarray): class scores
        iou_thr (float): IoU threshold. Defaults to 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: tp, fp, miou for each image
    """
    num_dets = len(det[0])  # M: number of prediction
    num_gts = len(gt_masks)  # N: number of gt

    tp = np.zeros(num_dets, dtype=np.float32)
    fp = np.zeros(num_dets, dtype=np.float32)
    gt_covered_iou = np.zeros(num_gts, dtype=np.float32)

    if len(gt_masks) == 0:
        fp[...] = 1
        return tp, fp, 0.0
    if num_dets == 0:
        return tp, fp, 0.0

    ious = mask_iou(det, gt_masks, iou_thr)  # (M, N)

    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sorted_inds = np.argsort(-cls_scores)

    gt_covered = np.zeros(num_gts, dtype=bool)
    # if no area range is specified, gt_area_ignore is all False
    for i in sorted_inds:
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


@METRICS.register_module()
class OTXDetMetric(VOCMetric):
    """OTX Object Detection Evaluator for mAP and mIoU."""

    nproc = 4

    def get_gt_instance_masks(self, annotations: list[dict]) -> list[list]:
        """Format ground truth instance mask annotation.

        Args:
            annotation (List[Dict]): per-image ground truth annotation

        Returns:
            cls_anno_list: per-class ground truth instance mask list
        """
        cls_anno_list: list[list] = [[] for _ in range(self.num_classes)]
        for class_id in range(self.num_classes):
            for annotation in annotations:
                gt_inds = annotation["labels"] == class_id
                polygon_masks = []
                if gt_inds.any():
                    gt_inds = np.where(gt_inds == 1)[0]
                    polygon_masks = annotation["masks"][gt_inds]
                cls_anno_list[class_id].append(polygon_masks)
        return cls_anno_list

    def get_mask_det_results(self, det_results: list[tuple], class_id: int) -> tuple[list, list]:
        """Get mask detection results for a specific class.

        Args:
            det_results (list(tuple)): detection results including bboxes and masks
            class_id (int): class index

        Returns:
            cls_dets: per-class detection results including bboxes and decoded masks
            cls_scores: class scores
        """
        cls_scores = [img_res[0][class_id][..., -1] for img_res in det_results]
        cls_dets: list[tuple] = []
        for det_result in det_results:
            det_bboxes = det_result[0][class_id][:, :4]
            det_masks = det_result[1][class_id]
            if len(det_masks) == 0:
                cls_dets.append(([], []))
            else:
                # Convert 28x28 encoded RLE mask detection to 28x28 BitmapMasks.
                det_masks = mask_util.decode(det_masks)
                det_masks = det_masks.transpose(2, 0, 1)
                det_masks = BitmapMasks(det_masks, *det_masks.shape[1:])
                cls_dets.append((det_bboxes, det_masks))
        return cls_dets, cls_scores

    def evaluate_mask(
        self,
        gts: list,
        preds: list,
        logger: logging.Logger,
        iou_thr: float,
    ) -> tuple[float, list[dict[str, Any]]]:
        """Evaluate mask results.

        Args:
            gts (list): list of ground truth
            preds(list): list of prediction
            logger (Logger): OTX logger
            iou_thr (float): IoU threshold

        Returns:
            metric (float): mAP and mIoU metric
            eval_results (list[dict[str, Any]]): Evaluation results
        """
        if len(gts) != len(preds):
            raise ValueDifferenceError(len(gts), len(preds))
        num_imgs = len(gts)
        eval_results = []

        gts = self.get_gt_masks(gts)

        ctx = mp.get_context("spawn")
        with ctx.Pool(self.nproc) as p:
            for class_id in range(len(self.dataset_meta["classes"])):
                # get gt and det bboxes of this class
                cls_dets, cls_scores = self.get_mask_det_results(preds, class_id)
                cls_gts = gts[class_id]

                # compute tp and fp for each image with multiple processes
                tpfpmiou = p.starmap(
                    compare_gts_and_predicitons,
                    zip(cls_dets, cls_gts, cls_scores, [iou_thr for _ in range(num_imgs)]),
                )
                tp, fp, miou = tuple(zip(*tpfpmiou))

                # sort all det bboxes by score, also sort tp and fp
                np_cls_scores = np.hstack(cls_scores)
                num_dets = np_cls_scores.shape[0]
                num_gts = np.sum([len(cls_gts) for cls_gts in cls_gts])
                sort_inds = np.argsort(np_cls_scores)[::-1]
                tp = np.hstack(tp)[sort_inds]
                fp = np.hstack(fp)[sort_inds]
                # calculate recall and precision with tp and fp
                tp = np.cumsum(tp)
                fp = np.cumsum(fp)
                eps = np.finfo(np.float32).eps
                recalls = tp / np.maximum(num_gts, eps)
                precisions = tp / np.maximum((tp + fp), eps)
                miou = np.mean(np.stack(miou))
                # calculate AP
                ap = average_precision(recalls, precisions, "area")
                eval_results.append(
                    {
                        "num_gts": num_gts,
                        "num_dets": num_dets,
                        "recall": recalls,
                        "precision": precisions,
                        "ap": ap,
                        "miou": miou,
                    },
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

        print_map_summary(mean_ap, eval_results, self.dataset_meta["classes"], None, logger=logger)

        return metrics["mAP"], eval_results

    def get_gt_masks(self, gts: list) -> list:
        """Return masks information from gt."""
        gt_masks = []
        for idx in range(len(self.dataset_meta["classes"])):
            gt_per_classes = []
            for gt in gts:
                gt_mask = gt["masks"][gt["labels"] == idx]
                if len(gt_mask) == 0:
                    gt_mask = []
                gt_per_classes.append(gt_mask)
            gt_masks.append(gt_per_classes)
        return gt_masks

    def compute_metrics(
        self,
        results: list,
    ) -> OrderedDict[str, float]:
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        eval_results = OrderedDict()
        gts, preds = zip(*results)
        mean_aps = []
        for iou_thr in self.iou_thrs:
            print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            if self.dataset_meta["domain"] == Domain.DETECTION:
                mean_ap, _ = eval_map(
                    preds,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.dataset_meta["classes"],
                    logger=logger,
                    eval_mode=self.eval_mode,
                    use_legacy_coordinate=True,
                )
            else:
                mean_ap, _ = self.evaluate_mask(gts, preds, logger, iou_thr)
            mean_aps.append(mean_ap)
            eval_results[f"AP{int(iou_thr * 100):02d}"] = round(mean_ap, 3)
        eval_results["mAP"] = sum(mean_aps) / len(mean_aps)
        return eval_results

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        del data_batch  # This variable is not used.

        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt["gt_instances"]
            gt_ignore_instances = gt["ignored_instances"]
            annotations = {
                "labels": gt_instances["labels"].cpu().numpy(),
                "bboxes": gt_instances["bboxes"].cpu().numpy(),
                "bboxes_ignore": gt_ignore_instances["bboxes"].cpu().numpy(),
                "labels_ignore": gt_ignore_instances["labels"].cpu().numpy(),
            }

            pred = data_sample["pred_instances"]
            pred_bboxes = pred["bboxes"].cpu().numpy()
            pred_scores = pred["scores"].cpu().numpy()
            pred_labels = pred["labels"].cpu().numpy()

            if self.dataset_meta["domain"] == Domain.INSTANCE_SEGMENTATION:
                annotations["masks"] = gt_instances["masks"]
                annotations["masks_ignore"] = gt_ignore_instances["masks"]
                pred_masks = np.array([encode_mask_results(mask)[0] for mask in pred["masks"]])

            detections = []
            masks = []
            for label in range(len(self.dataset_meta["classes"])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack([pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                detections.append(pred_bbox_scores)
                if self.dataset_meta["domain"] == Domain.INSTANCE_SEGMENTATION:
                    masks.append(pred_masks[index].tolist())

            if self.dataset_meta["domain"] == Domain.INSTANCE_SEGMENTATION:
                self.results.append((annotations, (detections, masks)))
            else:
                self.results.append((annotations, detections))
