# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX tile merge module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from datumaro import Bbox, DatasetItem, Image, Mask
from datumaro import Dataset as DmDataset
from datumaro.plugins.tiling.merge_tile import MergeTile
from torchvision import tv_tensors

from otx.core.data.entity.detection import DetBatchPredEntity, DetPredEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity, InstanceSegPredEntity

if TYPE_CHECKING:
    from otx.core.data.entity.base import ImageInfo


def merge_dataset_items(dataset_items: list[DatasetItem]) -> DatasetItem:
    """Merge dataset items into one single dataset item.

    Args:
        dataset_items (list[DatasetItem]): List of tile dataset items.

    Returns:
        DatasetItem: Merged dataset item.
    """
    dataset = DmDataset.from_iterable(dataset_items)
    return dataset.transform(MergeTile).get(next(iter(dataset)).id)


def create_merged_detection_prediction(img: np.ndarray, img_info: ImageInfo, merged_item: DatasetItem) -> DetPredEntity:
    """Create merged detection prediction.

    Args:
        img (np.ndarray): Original untiled image.
        img_info (ImageInfo): Image information.
        merged_item (DatasetItem): Merged dataset item.

    Returns:
        DetPredEntity: Merged detection prediction.
    """
    device = img_info.device
    pred_bboxes, pred_labels, pred_scores = [], [], []
    for anno in merged_item.annotations:
        if isinstance(anno, Bbox):
            pred_bboxes.append(anno.points)
            pred_labels.append(anno.label)
            pred_scores.append(float(anno.attributes["score"]))

    if len(pred_bboxes) == 0:
        pred_bboxes = torch.empty((0, 4))
    return DetPredEntity(
        image=tv_tensors.Image(img),
        img_info=img_info,
        score=torch.tensor(pred_scores, device=device),
        bboxes=tv_tensors.BoundingBoxes(pred_bboxes, format="XYXY", canvas_size=img.shape[:2], device=device),
        labels=torch.tensor(pred_labels, device=device),
    )


def create_merged_inst_seg_prediction(
    img: np.ndarray,
    img_info: ImageInfo,
    merged_item: DatasetItem,
) -> InstanceSegPredEntity:
    """Create merged instance segmentation prediction.

    Args:
        img (np.ndarray): Original untiled image.
        img_info (ImageInfo): Image information.
        merged_item (DatasetItem): Merged dataset item.

    Returns:
        InstanceSegPredEntity: Merged inst-seg prediction.
    """
    device = img_info.device
    pred_bboxes, pred_labels, pred_scores, pred_mask_instances = [], [], [], []
    pred_masks_by_label = {}
    for anno in merged_item.annotations:
        if isinstance(anno, Bbox):
            pred_bboxes.append(anno.points)
            pred_labels.append(anno.label)
            pred_scores.append(float(anno.attributes["score"]))
        if isinstance(anno, Mask):
            # NOTE: Datumaro tile merge does not give mask instances.
            # It merges mask instances to one mask.
            pred_masks_by_label[anno.label] = anno.image

    if len(pred_bboxes) == 0:
        pred_bboxes = torch.empty((0, 4))
    else:
        for pred_box, pred_label in zip(pred_bboxes, pred_labels):
            x1, y1, x2, y2 = (int(value) for value in pred_box)
            pred_label_mask = pred_masks_by_label[pred_label]
            # TODO (Eugene): Performance issue here if there are too many mask instances.
            # Ideally to optimize memory and speed we should convert it to RLE or save the crop pred_mask[y1:y2, x1:x2]
            # https://github.com/openvinotoolkit/datumaro/pull/1194
            bitmask = np.zeros_like(pred_label_mask)
            bitmask[y1:y2, x1:x2] = pred_label_mask[y1:y2, x1:x2]
            pred_mask_instances.append(bitmask)

    return InstanceSegPredEntity(
        image=tv_tensors.Image(img),
        img_info=img_info,
        score=torch.tensor(pred_scores, device=device),
        bboxes=tv_tensors.BoundingBoxes(pred_bboxes, format="XYXY", canvas_size=img.shape[:2], device=device),
        labels=torch.tensor(pred_labels, device=device),
        masks=tv_tensors.Mask(pred_mask_instances, dtype=torch.bool, device=device),
        polygons=[],
    )


def merge_detection_tiles(
    tile_preds: list[DetBatchPredEntity],
) -> DetPredEntity:
    """Merge detection tiles into one single detection prediction.

    Args:
        tile_preds (list[DetBatchPredEntity]): List of tile predictions.

    Returns:
        DetPredEntity: Merged detection prediction.
    """
    dataset_items = []
    anno_id = 0

    for tile_pred in tile_preds:
        annotations = []
        for batch_bboxes, batch_labels, batch_scores in zip(tile_pred.bboxes, tile_pred.labels, tile_pred.scores):
            for bbox, label, score in zip(batch_bboxes.data, batch_labels, batch_scores):
                _bbox = bbox.detach().cpu().numpy()
                _label = label.detach().cpu().numpy()
                _score = score.detach().cpu().numpy()
                x1, y1, x2, y2 = _bbox
                w, h = x2 - x1, y2 - y1
                annotations.append(
                    Bbox(x1, y1, w, h, label=_label, id=anno_id, attributes={"score": _score}),
                )
                anno_id += 1

        tile_info = tile_pred.imgs_info[0]
        tile_img = tile_pred.images[0].detach().cpu().numpy().transpose(1, 2, 0)
        tile_img = cv2.resize(tile_img, tile_info.ori_shape)
        dataset_item = DatasetItem(
            media=Image.from_numpy(tile_img),
            id=tile_info.attributes.get("tile_idx", 0),
            annotations=annotations,
            attributes=tile_info.attributes,
        )
        dataset_items.append(dataset_item)

    merged_item = merge_dataset_items(dataset_items)
    img_info = tile_preds[0].imgs_info[0]
    full_img = merged_item.media_as(Image).data
    return create_merged_detection_prediction(full_img, img_info, merged_item)


def merge_inst_seg_tiles(
    tile_preds: list[InstanceSegBatchPredEntity],
) -> InstanceSegPredEntity:
    """Merge instance segmentation tiles into one single inst-seg prediction.

    Args:
        tile_preds (list[InstanceSegBatchPredEntity]): List of tile inst-seg predictions.

    Returns:
        InstanceSegPredEntity: Merged inst-seg prediction.
    """
    dataset_items = []
    anno_id = 0
    for tile_pred in tile_preds:
        annotations = []
        for batch_bboxes, batch_labels, batch_scores, batch_masks in zip(
            tile_pred.bboxes,
            tile_pred.labels,
            tile_pred.scores,
            tile_pred.masks,
        ):
            for bbox, label, score, mask in zip(batch_bboxes.data, batch_labels, batch_scores, batch_masks):
                _bbox = bbox.detach().cpu().numpy()
                _label = label.detach().cpu().numpy()
                _score = score.detach().cpu().numpy()
                _mask = mask.detach().cpu().numpy()
                x1, y1, x2, y2 = _bbox
                w, h = x2 - x1, y2 - y1
                annotations.extend(
                    [
                        Mask(_mask, label=_label, id=anno_id, group=anno_id, attributes={"score": _score}),
                        Bbox(x1, y1, w, h, label=_label, id=anno_id, group=anno_id, attributes={"score": _score}),
                    ],
                )
                anno_id += 1

        tile_info = tile_pred.imgs_info[0]
        tile_img = tile_pred.images[0].detach().cpu().numpy().transpose(1, 2, 0)
        tile_img = cv2.resize(tile_img, tile_info.ori_shape)
        dataset_item = DatasetItem(
            media=Image.from_numpy(tile_img),
            id=tile_info.attributes["tile_idx"],
            annotations=annotations,
            attributes=tile_info.attributes,
        )
        dataset_items.append(dataset_item)

    merged_item = merge_dataset_items(dataset_items)
    img_info = tile_preds[0].imgs_info[0]
    full_img = merged_item.media_as(Image).data
    return create_merged_inst_seg_prediction(full_img, img_info, merged_item)
