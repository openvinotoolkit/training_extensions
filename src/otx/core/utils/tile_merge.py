# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX tile merge module."""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import Generic

import torch
from omegaconf import DictConfig
from torchvision import tv_tensors
from torchvision.ops import batched_nms

from otx.core.data.entity.base import ImageInfo, T_OTXBatchPredEntity, T_OTXDataEntity
from otx.core.data.entity.detection import DetBatchPredEntity, DetPredEntity
from otx.core.data.entity.instance_segmentation import (
    InstanceSegBatchPredEntity,
    InstanceSegPredEntity,
)


def search_by_key(d: DictConfig, key: str) -> DictConfig:
    """Search a key in a nested dictionary."""
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, DictConfig):
            value = search_by_key(v, key)
            if key in value:
                return value
    return DictConfig({})


class TileMerge(Generic[T_OTXDataEntity, T_OTXBatchPredEntity]):
    """Base class for tile merge.

    Args:
        img_infos (list[ImageInfo]): Original image information before tiling.
        iou_threshold (float, optional): IoU threshold for non-maximum suppression. Defaults to 0.45.
        max_num_instances (int, optional): Maximum number of instances to keep. Defaults to 100.

    TODO (Eugene): Find a way to configure tile merge parameters(score_thres, max_num, etc) from tile config.
    # https://github.com/openvinotoolkit/datumaro/pull/1194
    """

    def __init__(
        self,
        img_infos: list[ImageInfo],
        iou_threshold: float = 0.45,
        max_num_instances: int = 100,
    ) -> None:
        self.img_infos = img_infos
        self.iou_threshold = iou_threshold
        self.max_num_instances = max_num_instances

    @abstractmethod
    def _merge_entities(self, img_info: ImageInfo, entities: list[T_OTXDataEntity]) -> T_OTXDataEntity:
        """Merge tile predictions to one single full-size prediction data entity.

        Args:
            img_info (ImageInfo): Image information about the original image before tiling.
            entities (list[T_OTXDataEntity]): List of tile prediction entities.

        Returns:
            T_OTXDataEntity: Merged prediction entity.
        """
        raise NotImplementedError

    @abstractmethod
    def merge(
        self,
        batch_tile_preds: list[T_OTXBatchPredEntity],
        batch_tile_attrs: list[list[dict]],
    ) -> list[T_OTXDataEntity]:
        """Merge batch tile predictions to a list of full-size prediction data entities.

        Args:
            batch_tile_preds (list): list of tile predictions.
            batch_tile_attrs (list): list of tile attributes.
        """
        raise NotImplementedError

    def nms_postprocess(
        self,
        bboxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        masks: None | list[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None | torch.Tensor]:
        """Non-maximum suppression and post-process."""
        keep = batched_nms(bboxes, scores, labels, self.iou_threshold)
        if len(keep) > self.max_num_instances:
            keep = keep[: self.max_num_instances]
        bboxes = bboxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        if masks is not None and len(masks) > 0:
            masks = torch.stack([masks[idx] for idx in keep]).to_dense()
        return bboxes, labels, scores, masks

    @classmethod
    def from_config(cls, img_infos: list[ImageInfo], config: DictConfig) -> TileMerge:
        """Create a tile merge instance from a configuration.

        Args:
            img_infos (list[ImageInfo]): Original image information before tiling.
            config (DictConfig): Model configuration.

        Returns:
            TileMerge: Tile merge instance.
        """
        raise NotImplementedError


class DetectionTileMerge(TileMerge):
    """Detection tile merge."""

    @classmethod
    def from_config(cls, img_infos: list[ImageInfo], config: DictConfig) -> DetectionTileMerge:
        """Create a tile merge instance from a configuration."""
        test_cfg = search_by_key(config, "test_cfg")
        nms = search_by_key(test_cfg, "nms")
        iou_threshold = nms.get("iou_threshold", 0.45)
        max_num_instances = test_cfg.get("max_per_img", 100)

        return cls(img_infos, iou_threshold, max_num_instances)

    def merge(
        self,
        batch_tile_preds: list[DetBatchPredEntity],
        batch_tile_attrs: list[list[dict]],
    ) -> list[DetPredEntity]:
        """Merge batch tile predictions to a list of full-size prediction data entities.

        Args:
            batch_tile_preds (list): detection tile predictions.
            batch_tile_attrs (list): detection tile attributes.

        """
        entities_to_merge = defaultdict(list)
        img_ids = []

        for tile_preds, tile_attrs in zip(batch_tile_preds, batch_tile_attrs):
            for tile_attr, tile_img_info, tile_bboxes, tile_labels, tile_scores in zip(
                tile_attrs,
                tile_preds.imgs_info,
                tile_preds.bboxes,
                tile_preds.labels,
                tile_preds.scores,
            ):
                offset_x, offset_y, _, _ = tile_attr["roi"]
                tile_bboxes[:, 0::2] += offset_x
                tile_bboxes[:, 1::2] += offset_y

                tile_id = tile_attr["tile_id"]
                if tile_id not in img_ids:
                    img_ids.append(tile_id)
                tile_img_info.padding = tile_attr["roi"]

                entities_to_merge[tile_id].append(
                    DetPredEntity(
                        image=torch.empty(tile_img_info.ori_shape),
                        img_info=tile_img_info,
                        bboxes=tile_bboxes,
                        labels=tile_labels,
                        score=tile_scores,
                    ),
                )
        return [
            self._merge_entities(image_info, entities_to_merge[img_id])
            for img_id, image_info in zip(img_ids, self.img_infos)
        ]

    def _merge_entities(self, img_info: ImageInfo, entities: list[DetPredEntity]) -> DetPredEntity:
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
        for tile_entity in entities:
            num_preds = len(tile_entity.bboxes)
            if num_preds > 0:
                bboxes.extend(tile_entity.bboxes)
                labels.extend(tile_entity.labels)
                scores.extend(tile_entity.score)

        bboxes = torch.stack(bboxes) if len(bboxes) > 0 else torch.empty((0, 4), device=img_info.device)
        labels = torch.stack(labels) if len(labels) > 0 else torch.empty((0,), device=img_info.device)
        scores = torch.stack(scores) if len(scores) > 0 else torch.empty((0,), device=img_info.device)

        bboxes, labels, scores, _ = self.nms_postprocess(
            bboxes,
            scores,
            labels,
        )

        return DetPredEntity(
            image=torch.empty(img_size),
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

    @classmethod
    def from_config(cls, img_infos: list[ImageInfo], config: DictConfig) -> InstanceSegTileMerge:
        """Create a tile merge instance from a configuration."""
        test_cfg = search_by_key(config, "test_cfg")
        rcnn = search_by_key(test_cfg, "rcnn")
        nms = search_by_key(rcnn, "nms")

        iou_threshold = nms.get("iou_threshold", 0.45)
        max_num_instances = rcnn.get("max_per_img", 100)

        return cls(img_infos, iou_threshold, max_num_instances)

    def merge(
        self,
        batch_tile_preds: list[InstanceSegBatchPredEntity],
        batch_tile_attrs: list[list[dict]],
    ) -> list[InstanceSegPredEntity]:
        """Merge inst-seg tile predictions to one single prediction.

        Args:
            batch_tile_preds (list): instance-seg tile predictions.
            batch_tile_attrs (list): instance-seg tile attributes.

        """
        entities_to_merge = defaultdict(list)
        img_ids = []

        for tile_preds, tile_attrs in zip(batch_tile_preds, batch_tile_attrs):
            for tile_attr, tile_img_info, tile_bboxes, tile_labels, tile_scores, tile_masks in zip(
                tile_attrs,
                tile_preds.imgs_info,
                tile_preds.bboxes,
                tile_preds.labels,
                tile_preds.scores,
                tile_preds.masks,
            ):
                keep_indices = tile_masks.sum((1, 2)) > 0
                keep_indices = keep_indices.nonzero(as_tuple=True)[0]
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
                        image=torch.empty(tile_img_info.ori_shape),
                        img_info=tile_img_info,
                        bboxes=_bboxes,
                        labels=_labels,
                        score=_scores,
                        masks=_masks.to_sparse(),
                        polygons=[],
                    ),
                )

        return [
            self._merge_entities(image_info, entities_to_merge[img_id])
            for img_id, image_info in zip(img_ids, self.img_infos)
        ]

    def _merge_entities(self, img_info: ImageInfo, entities: list[InstanceSegPredEntity]) -> InstanceSegPredEntity:
        """Merge tile predictions to one single prediction.

        Args:
            img_info (ImageInfo): Image information about the original image before tiling.
            entities (list[InstanceSegPredEntity]): List of tile prediction entities.

        Returns:
            InstanceSegPredEntity: Merged prediction entity.
        """
        bboxes: list | torch.Tensor = []
        labels: list | torch.Tensor = []
        scores: list | torch.Tensor = []
        masks: list | torch.Tensor = []
        img_size = img_info.ori_shape
        for tile_entity in entities:
            num_preds = len(tile_entity.bboxes)
            if num_preds > 0:
                bboxes.extend(tile_entity.bboxes)
                labels.extend(tile_entity.labels)
                scores.extend(tile_entity.score)

                offset_x, offset_y, _, _ = tile_entity.img_info.padding
                mask_indices = tile_entity.masks.indices()
                mask_values = tile_entity.masks.values()
                mask_indices[1] += offset_y
                mask_indices[2] += offset_x
                masks.extend(
                    torch.sparse_coo_tensor(mask_indices, mask_values, (num_preds, *img_size)),
                )

        bboxes = torch.stack(bboxes) if len(bboxes) > 0 else torch.empty((0, 4), device=img_info.device)
        labels = torch.stack(labels) if len(labels) > 0 else torch.empty((0,), device=img_info.device)
        scores = torch.stack(scores) if len(scores) > 0 else torch.empty((0,), device=img_info.device)
        masks = masks if len(masks) > 0 else torch.empty((0, *img_size))

        bboxes, labels, scores, masks = self.nms_postprocess(bboxes, scores, labels, masks)
        return InstanceSegPredEntity(
            image=torch.empty(img_size),
            img_info=img_info,
            score=scores,
            bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                canvas_size=img_size,
                format="XYXY",
            ),
            labels=labels,
            masks=tv_tensors.Mask(masks, dtype=bool),
            polygons=[],
        )
