# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX tile merge module."""

from __future__ import annotations

import cv2
import torch
from datumaro import Bbox, DatasetItem, Image
from datumaro import Dataset as DmDataset
from datumaro.plugins.tiling.merge_tile import MergeTile
from torchvision import tv_tensors

from otx.core.data.entity.detection import DetBatchPredEntity, DetPredEntity


def merge(tile_preds: list[DetBatchPredEntity]) -> DetPredEntity:
    """Merge tile predictions into full image prediction.

    Args:
        tile_preds (list[OTXBatchPredEntity]): List of tile predictions

    Returns:
        OTXPredEntity: Full image prediction
    """
    dataset_items = []
    anno_id = 0
    for tile_pred in tile_preds:
        annotations = []
        tile_info = tile_pred.imgs_info[0]
        tile_img = tile_pred.images[0].detach().cpu().numpy().transpose(1, 2, 0)
        tile_img = cv2.resize(tile_img, tile_info.ori_shape)
        if len(tile_pred.bboxes) and len(tile_pred.bboxes[0]):
            # how to filter duplicated bbox using NMS?
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

        dataset_item = DatasetItem(
            media=Image.from_numpy(tile_img),
            id=tile_info.attributes["tile_idx"],
            annotations=annotations,
            attributes=tile_info.attributes,
        )
        dataset_items.append(dataset_item)

    pred_bboxes = []
    pred_labels = []
    pred_scores = []

    dataset = DmDataset.from_iterable(dataset_items)
    dataset = dataset.transform(MergeTile)
    ds_id = next(ds_item.id for ds_item in dataset)
    ds_item = dataset.get(ds_id)
    full_img = ds_item.media_as(Image).data

    for anno in ds_item.annotations:
        if isinstance(anno, Bbox):
            pred_bboxes.append(anno.points)
            pred_labels.append(anno.label)
            pred_scores.append(anno.attributes["score"])

    if len(pred_bboxes) == 0:
        pred_bboxes = torch.empty((0, 4))

    return DetPredEntity(
        image=tv_tensors.Image(full_img),
        img_info=tile_info,
        score=torch.tensor(pred_scores, device="cuda"),
        bboxes=tv_tensors.BoundingBoxes(pred_bboxes, format="XYXY", canvas_size=full_img.shape[:2], device="cuda"),
        labels=torch.tensor(pred_labels, device="cuda"),
    )
