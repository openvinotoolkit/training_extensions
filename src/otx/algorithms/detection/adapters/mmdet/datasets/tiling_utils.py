"""Tiling utils."""
import numpy as np
import torch
from mmcv.ops import nms
from mmrotate import obb2poly_np


def multiclass_nms(boxes: np.ndarray, scores: np.ndarray, idxs: np.ndarray, iou_threshold: float, max_num: int):
    """NMS for multi-class bboxes.

    Args:
        boxes (np.ndarray):  boxes in shape (N, 4).
        scores (np.ndarray): scores in shape (N, ).
        idxs (np.ndarray):  each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        iou_threshold (float): IoU threshold to be used to suppress boxes
            in tiles' overlap areas.
        max_num (int): if there are more than max_per_img bboxes after
            NMS, only top max_per_img will be kept.

    Returns:
        tuple: tuple: kept dets and indice.
    """
    if len(boxes) == 0:
        return None, []
    max_coordinate = boxes.max()
    offsets = idxs.astype(boxes.dtype) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    dets, keep = nms(boxes_for_nms, scores, iou_threshold)
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    return dets, keep


def tile_boxes_overlap(tile_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute overlapping ratio over boxes.

    Args:
        tile_box (np.ndarray): box in shape (1, 4).
        boxes (np.ndarray): boxes in shape (N, 4).

    Returns:
        np.ndarray: matched indices.
    """
    x1, y1, x2, y2 = tile_box[0]
    match_indices = (boxes[:, 0] > x1) & (boxes[:, 1] > y1) & (boxes[:, 2] < x2) & (boxes[:, 3] < y2)
    match_indices = np.argwhere(match_indices == 1).flatten()
    return match_indices


def tile_rboxes_overlap(tile_box: np.ndarray, rboxes: np.ndarray, angle_version: str) -> np.ndarray:
    """Compute overlapping ratio over rotated boxes.

    Args:
        tile_box (np.ndarray): box in shape (1, 4).
        rboxes (np.ndarray): rotated boxes in shape (N, 5) as in cx, cy, w, h, a.
        angle_version (str): angle version of "le90", "le135", or "oc"

    Returns:
        np.ndarray: matched indices.
    """
    x1, y1, x2, y2 = tile_box[0]
    if len(rboxes) == 0:
        return np.array([], dtype=np.int64)
    # rbox cx, cy, w, h, a, score
    polygons = obb2poly_np(np.concatenate((rboxes, np.ones((rboxes.shape[0], 1))), axis=1), angle_version)
    polygons = polygons[:, :-1]
    dummy_boxes = np.concatenate(
        (
            np.min(polygons[:, 0::2], axis=1, keepdims=True),
            np.min(polygons[:, 1::2], axis=1, keepdims=True),
            np.max(polygons[:, 0::2], axis=1, keepdims=True),
            np.max(polygons[:, 1::2], axis=1, keepdims=True),
        ),
        axis=1,
    )
    match_indices = (
        (dummy_boxes[:, 0] > x1) & (dummy_boxes[:, 1] > y1) & (dummy_boxes[:, 2] < x2) & (dummy_boxes[:, 3] < y2)
    )
    match_indices = np.argwhere(match_indices == 1).flatten()
    return match_indices


def translate_boxes(tile_bboxes, shift_x, shift_y):
    """Shift boxes by shift_x and shift_y and clip to tile_edge_size.

    Args:
        tile_bboxes (np.array): boxes in shape (N, 4).
        shift_x (int): shift in x direction.
        shift_y (int): shift in y direction.

    Returns:
        tile_bboxes (np.array): shifted boxes in shape (N, 4).
    """
    tile_bboxes[:, 0] += shift_x
    tile_bboxes[:, 1] += shift_y
    tile_bboxes[:, 2] += shift_x
    tile_bboxes[:, 3] += shift_y
    return tile_bboxes


def translate_rboxes(tile_rboxes, shift_x, shift_y):
    """Shift rotated boxes by shift_x and shift_y and clip to tile_edge_size.

    Args:
        tile_rboxes (np.array): rboxes in shape (N, 5) as in cx, cy, w, h, a.
        shift_x (int): shift in x direction.
        shift_y (int): shift in y direction.
        angle_version (str): angle version.

    Returns:
        tile_rboxes (np.array): shifted rboxes in shape (N, 5) as in cx, cy, w, h, a.
    """
    tile_rboxes[:, 0] += shift_x
    tile_rboxes[:, 1] += shift_y
    return tile_rboxes


def rbbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): shape (n, 6)
        labels (torch.Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for _ in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]
