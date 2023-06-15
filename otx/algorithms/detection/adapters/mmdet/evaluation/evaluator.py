from typing import List, Dict, Tuple
from multiprocessing import Pool
import numpy as np
import pycocotools.mask as mask_util
from mmdet.core.evaluation.mean_ap import average_precision
from otx.api.entities.label import Domain
from mmdet.core import eval_map, BitmapMasks, PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from .mean_ap_seg import print_map_summary


def display_bitmask(bitmask):
    import matplotlib.pyplot as plt
    plt.imshow(bitmask.astype(np.uint8))
    plt.show()


def sanitize_coordinates(bbox: np.ndarray, height: int, width: int, padding=1) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.int)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return np.array([x1, y1, x2, y2])


def mask_iou(det: Tuple[np.ndarray, BitmapMasks], gt_masks: PolygonMasks) -> np.ndarray:
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
        # expand det_mask and gt_mask to the same size
        min_x1 = min(det_bbox[0], gt_bbox[0])
        min_y1 = min(det_bbox[1], gt_bbox[1])
        max_x2 = max(det_bbox[2], gt_bbox[2])
        max_y2 = max(det_bbox[3], gt_bbox[3])
        det_bbox_h, det_bbox_w = det_bbox[3] - det_bbox[1], det_bbox[2] - det_bbox[0]
        det_mask = det_mask.resize((det_bbox_h, det_bbox_w))
        det_mask = det_mask.expand(max_y2 - min_y1, max_x2 - min_x1, det_bbox[1] - min_y1, det_bbox[0] - min_x1)
        gt_mask = gt_mask.crop(gt_bbox)
        gt_mask: BitmapMasks = gt_mask.to_bitmap()
        gt_mask = gt_mask.expand(max_y2 - min_y1, max_x2 - min_x1, gt_bbox[1] - min_y1, gt_bbox[0] - min_x1)
        # compute iou between det_mask and gt_mask
        det_mask = det_mask.to_ndarray()
        gt_mask = gt_mask.to_ndarray()
        assert det_mask.shape == gt_mask.shape, f"det_mask.shape={det_mask.shape} != gt_mask.shape={gt_mask.shape}"
        ious[m, n] = np.sum(det_mask & gt_mask) / np.sum(det_mask | gt_mask)
    return ious


def tpfpmiou_func(  # pylint: disable=too-many-locals
    det: Tuple[np.ndarray, BitmapMasks],
    gt_masks: List[Dict],
    cls_scores,
    iou_thr=0.5
):
    num_dets = len(det[0])  # M
    num_gts = len(gt_masks)    # N

    tp = np.zeros(num_dets, dtype=np.float32)  # pylint: disable=invalid-name
    fp = np.zeros(num_dets, dtype=np.float32)  # pylint: disable=invalid-name
    gt_covered_iou = np.zeros(num_gts, dtype=np.float32)

    if len(gt_masks) == 0:
        fp[...] = 1
        return tp, fp, 0.0
    if num_dets == 0:
        return tp, fp, 0.0

    ious = mask_iou(det, gt_masks)  # (M, N)

    assert ious.shape[0] == num_dets, f"{ious.shape[0]} != {num_dets}"
    assert ious.shape[1] == num_gts, f"{ious.shape[1]} != {num_gts}"

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
    def __init__(self, annotation, domain, classes, nproc=4):
        self.domain = domain
        self.classes = classes
        self.num_classes = len(classes)
        if domain != Domain.DETECTION:
            self.annotation = self.get_gt_instance_masks(annotation)
        else:
            self.annotation = annotation
        self.nproc = nproc

    def get_gt_instance_masks(self, annotation):
        """Crop mask from original image and saved them as PolygonMask"""
        num_images = len(annotation)
        cls_anno_list = [[] for _ in range(self.num_classes)]
        for idx in range(num_images):
            masks = annotation[idx]["masks"]
            for class_id in range(self.num_classes):
                gt_inds = annotation[idx]["labels"] == class_id
                polygon_masks = []
                if gt_inds.any():
                    gt_inds = np.where(gt_inds == 1)[0]
                    polygon_masks = masks[gt_inds]
                cls_anno_list[class_id].append(polygon_masks)
        return cls_anno_list

    def get_mask_det_results(self, det_results, class_id) -> Tuple[List, List]:
        cls_scores = [img_res[0][class_id][..., -1] for img_res in det_results]
        cls_dets = []
        for i, det in enumerate(det_results):
            det_bboxes = det[0][class_id][:, :4]
            det_masks = det[1][class_id]
            if len(det_masks) == 0:
                cls_dets.append([[] for _ in range(2)])
            else:
                det_masks = mask_util.decode(det_masks)
                det_masks = det_masks.transpose(2, 0, 1)
                det_masks = BitmapMasks(det_masks, *det_masks.shape[1:])
                cls_dets.append((det_bboxes, det_masks))
        return cls_dets, cls_scores

    def evaluate_mask(self, results, logger, iou_thr):
        assert len(results) == len(self.annotation[0]), "number of images should be equal!"
        num_imgs = len(results)
        pool = Pool(self.nproc)  # pylint: disable=consider-using-with
        eval_results = []
        for class_id in range(self.num_classes):
            # get gt and det bboxes of this class
            cls_dets, cls_scores = self.get_mask_det_results(results, class_id)
            cls_gts = self.annotation[class_id]

            # compute tp and fp for each image with multiple processes
            tpfpmiou = pool.starmap(tpfpmiou_func, zip(cls_dets, cls_gts, cls_scores, [iou_thr for _ in range(num_imgs)]))
            tp, fp, miou = tuple(zip(*tpfpmiou))  # pylint: disable=invalid-name

            # for loop version
            # tp, fp, miou = [], [], []
            # for i in range(num_imgs):
            #     tpfpmiou = tpfpmiou_func(cls_dets[i], cls_gts[i], cls_scores[i], iou_thr)
            #     tp.append(tpfpmiou[0])
            #     fp.append(tpfpmiou[1])
            #     miou.append(tpfpmiou[2])

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

        print_map_summary(mean_ap, eval_results, self.classes, None, logger=logger)

        return metrics['mAP'], eval_results

    def evaluate(self, results, logger, iou_thr, scale_ranges):
        if self.domain == Domain.DETECTION:
            return eval_map(
                    results,
                    self.annotation,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.classes,
                    logger=logger)
        return self.evaluate_mask(results, logger, iou_thr)
