from multiprocessing import Pool
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pycocotools.mask as mask_util
from mmdet.core.evaluation.mean_ap import average_precision
from otx.api.entities.label import Domain
from mmdet.core import eval_map

from .mean_ap_seg import print_map_summary, tpfpmiou_func


def _do_paste_mask_cpu(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (ndarray): N, 1, H, W
        boxes (ndarray): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (ndarray, tuple). The first item is mask ndarray, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    if skip_empty:
        x0_int, y0_int = np.clip(
            np.floor(np.min(boxes, axis=0)[:2]) - 1,
            0, None).astype(np.int32)
        x1_int = np.clip(
            np.ceil(np.max(boxes[:, 2])) + 1, None, img_w).astype(np.int32)
        y1_int = np.clip(
            np.ceil(np.max(boxes[:, 3])) + 1, None, img_h).astype(np.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = np.split(boxes, 4, 1)  # each is Nx1

    N = masks.shape[0]

    img_y = np.arange(y0_int, y1_int).astype(np.float32) + 0.5
    img_x = np.arange(x0_int, x1_int).astype(np.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    if np.isinf(img_x).any():
        inds = np.where(np.isinf(img_x))
        img_x[inds] = 0
    if np.isinf(img_y).any():
        inds = np.where(np.isinf(img_y))
        img_y[inds] = 0

    gx = np.expand_dims(img_x, axis=1).repeat(img_y.size, axis=1)
    gy = np.expand_dims(img_y, axis=2).repeat(img_x.size, axis=2)
    grid = np.stack([gx, gy], axis=3)

    img_masks = RegularGridInterpolator(grid, masks.astype(np.float32), bounds_error=False, fill_value=None)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


class Evaluator:
    def __init__(self, annotation, domain, img_metas, classes, mask_canvas=(800, 800), nproc=4):
        self.domain = domain
        self.img_metas = img_metas
        self.classes = classes
        self.num_classes = len(classes)
        self.mask_canvas = mask_canvas
        if domain != Domain.DETECTION:
            self.annotation = self.init_gt_instnace_masks(annotation)
        else:
            self.annotation = annotation
        self.nproc = nproc

    def init_gt_instnace_masks(self, annotation):
        num_images = len(annotation)
        cls_anno_list = [[] for _ in range(self.num_classes)]
        canvas_h, canvas_w = self.mask_canvas
        for idx in range(num_images):
            masks = annotation[idx]["masks"]
            masks = masks.resize((canvas_h, canvas_w)).to_ndarray()

            for class_id in range(self.num_classes):
                gt_inds = annotation[idx]["labels"] == class_id
                encoded_masks = []
                if gt_inds.any():
                    class_masks = masks[gt_inds]
                    encoded_masks = [
                        mask_util.encode(
                            np.array(m[:, :, np.newaxis], order="F", dtype="uint8")
                        )[0] for m in class_masks
                    ]
                cls_anno_list[class_id].append(encoded_masks)
        return cls_anno_list

    def evaluate(self, results, logger, iou_thr, scale_ranges):
        if self.domain == Domain.DETECTION:
            return eval_map(
                    results,
                    self.annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.classes,
                    logger=logger)
        return self.evaluate_mask(results, logger, iou_thr)

    def get_mask_det_results(self, det_results, class_id, img_metas):
        canvas_h, canvas_w = self.mask_canvas
        cls_scores = [img_res[0][class_id][..., -1] for img_res in det_results]
        cls_dets = []
        for i, det in enumerate(det_results):
            det_masks = det[1][class_id]
            det_bboxes = det[0][class_id][:, :4]

            if len(det_masks) == 0:
                det_masks = np.zeros((0, canvas_h, canvas_w), dtype=np.uint8)
            else:
                img_h, img_w = img_metas[i]['ori_shape'][:2]
                new_w_scale, new_h_scale = canvas_w/img_w, canvas_h/img_h
                scale_factor = np.tile([new_w_scale, new_h_scale], 2)
                det_masks = mask_util.decode(det_masks)
                det_masks = det_masks.transpose(2, 0, 1)
                det_masks = det_masks[:, np.newaxis]
                det_bboxes = det_bboxes * scale_factor
                det_masks, _ = _do_paste_mask_cpu(
                    det_masks,
                    det_bboxes,
                    canvas_h,
                    canvas_w,
                    skip_empty=False
                )
                det_masks = det_masks.numpy()
            det_masks = [
                mask_util.encode(np.array(m[:, :, np.newaxis], order="F", dtype="uint8"))[0] for m in det_masks
            ]
            cls_dets.append(det_masks)
        return cls_dets, cls_scores

    def evaluate_mask(self, results, logger, iou_thr):
        assert len(results) == len(self.annotation[0])

        num_imgs = len(results)

        pool = Pool(self.nproc)  # pylint: disable=consider-using-with
        eval_results = []
        for class_id in range(self.num_classes):
            # get gt and det bboxes of this class
            cls_dets, cls_scores = self.get_mask_det_results(results, class_id, self.img_metas)
            cls_gts = self.annotation[class_id]

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