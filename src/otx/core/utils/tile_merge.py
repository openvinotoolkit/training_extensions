# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX tile merge module."""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import Generic

import cv2
import numpy as np
import torch
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo, T_OTXDataEntity
from otx.core.data.entity.detection import DetBatchPredEntity, DetPredEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity, InstanceSegPredEntity


class TileMerge(Generic[T_OTXDataEntity]):
    """Base class for tile merge.

    Args:
        img_infos (list[ImageInfo]): Original image information before tiling.
        score_thres (float, optional): Score threshold to filter out low score predictions. Defaults to 0.1.
        max_num_instances (int, optional): Maximum number of instances to keep. Defaults to 100.

    TODO (Eugene): Find a way to configure tile merge parameters(score_thres, max_num, etc) from tile config.
    # https://github.com/openvinotoolkit/datumaro/pull/1194
    """

    def __init__(
        self,
        img_infos: list[ImageInfo],
        score_thres: float = 0.25,
        max_num_instances: int = 100,
    ) -> None:
        self.img_infos = img_infos
        self.score_thres = score_thres
        self.max_num_instances = max_num_instances

    @abstractmethod
    def merge_entities(self, img_info: ImageInfo, entities: list[T_OTXDataEntity]) -> T_OTXDataEntity:
        """Merge tile predictions to one single prediction.

        Args:
            img_info (ImageInfo): Image information about the original image before tiling.
            entities (list[T_OTXDataEntity]): List of tile prediction entities.

        Returns:
            T_OTXDataEntity: Merged prediction entity.
        """
        raise NotImplementedError

    @abstractmethod
    def merge(self, batch_tile_preds: list, batch_tile_attrs: list) -> list[T_OTXDataEntity]:
        """Merge tile predictions to one single prediction.

        Args:
            batch_tile_preds (list): list of tile predictions.
            batch_tile_attrs (list): list of tile attributes.
        """
        raise NotImplementedError


class DetectionTileMerge(TileMerge):
    """Detection tile merge."""

    def merge(self, batch_tile_preds: list[DetBatchPredEntity], batch_tile_attrs: list) -> list[DetPredEntity]:
        """Merge detection tile predictions to one single prediction.

        Args:
            batch_tile_preds (list): detection tile predictions.
            batch_tile_attrs (list): detection tile attributes.

        """
        entities_to_merge = defaultdict(list)
        img_ids = []

        for tile_preds, tile_attrs in zip(batch_tile_preds, batch_tile_attrs):
            for tile_attr, tile_img, tile_img_info, tile_bboxes, tile_labels, tile_scores in zip(
                tile_attrs,
                tile_preds.images,
                tile_preds.imgs_info,
                tile_preds.bboxes,
                tile_preds.labels,
                tile_preds.scores,
            ):
                keep_indices = tile_scores > self.score_thres
                keep_indices = keep_indices.nonzero(as_tuple=True)[0]
                _tile_img = tile_img.detach().cpu().numpy()
                _tile_img = cv2.resize(_tile_img.transpose(1, 2, 0), tile_img_info.ori_shape)
                _bboxes = tile_bboxes[keep_indices]
                _labels = tile_labels[keep_indices]
                _scores = tile_scores[keep_indices]

                offset_x, offset_y, _, _ = tile_attr["roi"]
                _bboxes[:, 0::2] += offset_x
                _bboxes[:, 1::2] += offset_y

                tile_id = tile_attr["tile_id"]
                if tile_id not in img_ids:
                    img_ids.append(tile_id)
                tile_img_info.padding = tile_attr["roi"]

                entities_to_merge[tile_id].append(
                    DetPredEntity(
                        image=_tile_img,
                        img_info=tile_img_info,
                        bboxes=_bboxes,
                        labels=_labels,
                        score=_scores,
                    ),
                )

        predictions = []
        for img_id, image_info in zip(img_ids, self.img_infos):
            predictions.append(self.merge_entities(image_info, entities_to_merge[img_id]))
        return predictions

    def merge_entities(self, img_info: ImageInfo, entities: list[DetPredEntity]) -> DetPredEntity:
        """Merge tile predictions to one single prediction.

        Args:
            img_info (ImageInfo): Image information about the original image before tiling.
            entities (list[DetPredEntity]): List of tile prediction entities.

        Returns:
            DetPredEntity: Merged prediction entity.
        """
        bboxes: list | torch.Tensor = []
        labels: list | torch.Tensor = []
        scores: list | torch.Tensor = []
        img_size = img_info.ori_shape
        full_img = np.zeros((*img_size, 3))
        for tile_entity in entities:
            num_preds = len(tile_entity.bboxes)
            tile_img = tile_entity.image
            tile_img_info = tile_entity.img_info
            x1, y1, w, h = tile_img_info.padding
            full_img[y1 : y1 + h, x1 : x1 + w] = tile_img
            if num_preds > 0:
                bboxes.extend(tile_entity.bboxes)
                labels.extend(tile_entity.labels)
                scores.extend(tile_entity.score)

        bboxes = torch.stack(bboxes) if len(bboxes) > 0 else torch.empty((0, 4), device=img_info.device)
        labels = torch.stack(labels) if len(labels) > 0 else torch.empty((0,), device=img_info.device)
        scores = torch.stack(scores) if len(scores) > 0 else torch.empty((0,), device=img_info.device)

        sort_inds = torch.argsort(scores, descending=True)
        if len(sort_inds) > self.max_num_instances:
            sort_inds = sort_inds[: self.max_num_instances]
        bboxes = bboxes[sort_inds]
        labels = labels[sort_inds]
        scores = scores[sort_inds]

        return DetPredEntity(
            image=full_img,
            img_info=img_info,
            score=scores,
            bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                canvas_size=img_size,
                format="XYXY",
            ),
            labels=labels,
        )


class InstanceSegTileMerge(TileMerge):
    """Instance segmentation tile merge."""

    def merge(
        self,
        batch_tile_preds: list[InstanceSegBatchPredEntity],
        batch_tile_attrs: list,
    ) -> list[InstanceSegPredEntity]:
        """Merge inst-seg tile predictions to one single prediction.

        Args:
            batch_tile_preds (list): instance-seg tile predictions.
            batch_tile_attrs (list): instance-seg tile attributes.

        """
        entities_to_merge = defaultdict(list)
        img_ids = []

        for tile_preds, tile_attrs in zip(batch_tile_preds, batch_tile_attrs):
            for tile_attr, tile_img, tile_img_info, tile_bboxes, tile_labels, tile_scores, tile_masks in zip(
                tile_attrs,
                tile_preds.images,
                tile_preds.imgs_info,
                tile_preds.bboxes,
                tile_preds.labels,
                tile_preds.scores,
                tile_preds.masks,
            ):
                keep_indices = tile_scores > self.score_thres
                keep_indices = keep_indices.nonzero(as_tuple=True)[0]
                _tile_img = tile_img.detach().cpu().numpy()
                _tile_img = cv2.resize(_tile_img.transpose(1, 2, 0), tile_img_info.ori_shape)
                _bboxes = tile_bboxes[keep_indices]
                _labels = tile_labels[keep_indices]
                _scores = tile_scores[keep_indices]
                _masks = tile_masks[keep_indices]

                offset_x, offset_y, _, _ = tile_attr["roi"]
                _bboxes[:, 0::2] += offset_x
                _bboxes[:, 1::2] += offset_y

                tile_id = tile_attr["tile_id"]
                if tile_id not in img_ids:
                    img_ids.append(tile_id)
                tile_img_info.padding = tile_attr["roi"]

                entities_to_merge[tile_id].append(
                    InstanceSegPredEntity(
                        image=_tile_img,
                        img_info=tile_img_info,
                        bboxes=_bboxes,
                        labels=_labels,
                        score=_scores,
                        masks=_masks.to_sparse(),
                        polygons=[],
                    ),
                )

        predictions = []
        for img_id, image_info in zip(img_ids, self.img_infos):
            predictions.append(self.merge_entities(image_info, entities_to_merge[img_id]))
        return predictions

    def merge_entities(self, img_info: ImageInfo, entities: list[InstanceSegPredEntity]) -> InstanceSegPredEntity:
        """Merge tile predictions to one single prediction.

        Args:
            img_info (ImageInfo): Image information about the original image before tiling.
            entities (list[InstanceSegPredEntity]): List of tile prediction entities.

        Returns:
            InstanceSegPredEntity: Merged prediction entity.
        """
        device = img_info.device
        bboxes = torch.tensor([], device=device)
        labels = torch.tensor([], device=device)
        scores = torch.tensor([], device=device)
        masks = []
        img_size = img_info.ori_shape
        full_img = np.zeros((*img_size, 3))
        for tile_entity in entities:
            num_preds = len(tile_entity.bboxes)
            tile_img = tile_entity.image
            tile_img_info = tile_entity.img_info
            x1, y1, w, h = tile_img_info.padding
            full_img[y1 : y1 + h, x1 : x1 + w] = tile_img
            if num_preds > 0:
                bboxes = torch.cat((bboxes, tile_entity.bboxes), dim=0)
                labels = torch.cat((labels, tile_entity.labels), dim=0)
                scores = torch.cat((scores, tile_entity.score), dim=0)
                sparse_masks_indices = tile_entity.masks.indices()
                masks_value = tile_entity.masks.values()
                if len(sparse_masks_indices):
                    sparse_masks_indices[1] += tile_img_info.padding[1]
                    sparse_masks_indices[2] += tile_img_info.padding[0]
                masks.extend(
                    torch.sparse_coo_tensor(sparse_masks_indices, masks_value, (num_preds, *img_size)),
                )

        sort_inds = torch.argsort(scores, descending=True)
        if len(sort_inds) > self.max_num_instances:
            sort_inds = sort_inds[: self.max_num_instances]
        bboxes = bboxes[sort_inds]
        labels = labels[sort_inds]
        scores = scores[sort_inds]
        masks = torch.stack([masks[idx] for idx in sort_inds]).to_dense() if len(masks) > 0 else []
        return InstanceSegPredEntity(
            image=full_img,
            img_info=img_info,
            score=scores,
            bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                canvas_size=img_size,
                format="XYXY",
            ),
            labels=labels,
            masks=masks,
            polygons=[],
        )
