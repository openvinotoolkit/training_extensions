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

from otx.core.data.entity.detection import DetBatchDataEntity, DetPredEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegPredEntity

if TYPE_CHECKING:
    from otx.core.data.entity.base import ImageInfo


def extract_det_preds(dataset_item: DatasetItem, img_info: ImageInfo, device: torch.device) -> DetPredEntity:
    full_img = dataset_item.media_as(Image).data
    pred_bboxes = []
    pred_labels = []
    pred_scores = []
    for anno in dataset_item.annotations:
        if isinstance(anno, Bbox):
            pred_bboxes.append(anno.points)
            pred_labels.append(anno.label)
            pred_scores.append(anno.attributes["score"])

    if len(pred_bboxes) == 0:
        pred_bboxes = torch.empty((0, 4))
    return DetPredEntity(
        image=tv_tensors.Image(full_img),
        img_info=img_info,
        score=torch.tensor(pred_scores, device=device),
        bboxes=tv_tensors.BoundingBoxes(pred_bboxes, format="XYXY", canvas_size=full_img.shape[:2], device=device),
        labels=torch.tensor(pred_labels, device=device),
    )


def extract_inst_seg_preds(
    dataset_item: DatasetItem,
    img_info: ImageInfo,
    device: torch.device,
) -> InstanceSegPredEntity:
    full_img = dataset_item.media_as(Image).data
    pred_bboxes = []
    pred_labels = []
    pred_scores = []
    pred_mask_instances = []
    pred_masks_by_label = {}
    for anno in dataset_item.annotations:
        if anno.attributes["score"] > 0.3:
            if isinstance(anno, Bbox):
                pred_bboxes.append(anno.points)
                pred_labels.append(anno.label)
                pred_scores.append(anno.attributes["score"])
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
            # TODO: Performance issue here if there are too many instances.
            # Add also copying mem to GPU is slow.
            bitmask = np.zeros_like(pred_label_mask)
            bitmask[y1:y2, x1:x2] = pred_label_mask[y1:y2, x1:x2]
            pred_mask_instances.append(bitmask)

    return InstanceSegPredEntity(
        image=tv_tensors.Image(full_img),
        img_info=img_info,
        score=torch.tensor(pred_scores, device=device),
        bboxes=tv_tensors.BoundingBoxes(pred_bboxes, format="XYXY", canvas_size=full_img.shape[:2], device=device),
        labels=torch.tensor(pred_labels, device=device),
        masks=tv_tensors.Mask(pred_mask_instances, dtype=torch.bool, device=device),
        polygons=[],
    )


def merge_detection_tiles(
    tile_preds: list[DetBatchDataEntity],
) -> DetPredEntity:
    dataset_items = []
    anno_id = 0
    for tile_pred in tile_preds:
        annotations = []
        if len(tile_pred.bboxes) and len(tile_pred.bboxes[0]):
            bboxes = tile_pred.bboxes[0].detach().cpu().numpy()
            labels = tile_pred.labels[0].detach().cpu().numpy()
            scores = tile_pred.scores[0].detach().cpu().numpy()
            for bbox, label, score in zip(bboxes, labels, scores):
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                annotations.append(
                    Bbox(x1, y1, w, h, label=label, id=anno_id, attributes={"score": score}),
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

    dataset = DmDataset.from_iterable(dataset_items)
    dataset = dataset.transform(MergeTile)
    ds_id = next(ds_item.id for ds_item in dataset)
    ds_item = dataset.get(ds_id)
    img_info = tile_preds[0].imgs_info[0]
    device = tile_preds[0].images[0].device
    return extract_det_preds(ds_item, img_info, device)


def merge_inst_seg_tiles(
    tile_preds: list[InstanceSegBatchDataEntity],
) -> InstanceSegPredEntity:
    dataset_items = []
    anno_id = 0
    for tile_pred in tile_preds:
        annotations = []
        if len(tile_pred.bboxes) and len(tile_pred.bboxes[0]):
            bboxes = tile_pred.bboxes[0].detach().cpu().numpy()
            labels = tile_pred.labels[0].detach().cpu().numpy()
            scores = tile_pred.scores[0].detach().cpu().numpy()
            masks = tile_pred.masks[0].detach().cpu().numpy()
            for bbox, label, score, mask in zip(bboxes, labels, scores, masks):
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                if mask[int(y1) : int(y2), int(x1) : int(x2)].sum() > 64:
                    annotations.extend(
                        [
                            Mask(mask, label=label, id=anno_id, group=anno_id, attributes={"score": score}),
                            Bbox(x1, y1, w, h, label=label, id=anno_id, group=anno_id, attributes={"score": score}),
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

    dataset = DmDataset.from_iterable(dataset_items)
    dataset = dataset.transform(MergeTile)
    ds_id = next(ds_item.id for ds_item in dataset)
    ds_item = dataset.get(ds_id)
    img_info = tile_preds[0].imgs_info[0]
    device = tile_preds[0].images[0].device

    return extract_inst_seg_preds(ds_item, img_info, device)
