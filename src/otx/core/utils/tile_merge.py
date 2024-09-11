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
from torchvision.ops import batched_nms

from otx.algo.explain.explain_algo import InstSegExplainAlgo
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import ImageInfo, T_OTXBatchPredEntity, T_OTXDataEntity
from otx.core.data.entity.detection import DetBatchPredEntity, DetPredEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity, InstanceSegPredEntity


class TileMerge(Generic[T_OTXDataEntity, T_OTXBatchPredEntity]):
    """Base class for tile merge.

    Args:
        img_infos (list[ImageInfo]): Original image information before tiling.
        num_classes (int): Number of classes.
        tile_config (TileConfig): Tile configuration.
        explain_mode (bool, optional): Whether or not tiles have explain features. Default: False.
    """

    def __init__(
        self,
        img_infos: list[ImageInfo],
        num_classes: int,
        tile_config: TileConfig,
        explain_mode: bool = False,
    ) -> None:
        self.img_infos = img_infos
        self.num_classes = num_classes
        self.tile_size = tile_config.tile_size
        self.iou_threshold = tile_config.iou_threshold
        self.max_num_instances = tile_config.max_num_instances
        self.with_full_img = tile_config.with_full_img
        self.explain_mode = explain_mode

    @abstractmethod
    def _merge_entities(
        self,
        img_info: ImageInfo,
        entities: list[T_OTXDataEntity],
        explain_mode: bool = False,
    ) -> T_OTXDataEntity:
        """Merge tile predictions to one single full-size prediction data entity.

        Args:
            img_info (ImageInfo): Image information about the original image before tiling.
            entities (list[T_OTXDataEntity]): List of tile prediction entities.
            explain_mode (bool): Whether or not tiles have explain features. Default: False.

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
            # coalesce sparse tensors to prevent them from growing too large.
            masks = torch.stack([masks[idx] for idx in keep]).coalesce().to_dense()
        return bboxes, labels, scores, masks


class DetectionTileMerge(TileMerge):
    """Detection tile merge."""

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
        explain_mode = self.explain_mode

        for tile_preds, tile_attrs in zip(batch_tile_preds, batch_tile_attrs, strict=True):
            batch_size = len(tile_attrs)
            saliency_maps = tile_preds.saliency_map if explain_mode else [[] for _ in range(batch_size)]
            feature_vectors = tile_preds.feature_vector if explain_mode else [[] for _ in range(batch_size)]
            for tile_attr, tile_img_info, tile_bboxes, tile_labels, tile_scores, tile_s_map, tile_f_vect in zip(
                tile_attrs,
                tile_preds.imgs_info,
                tile_preds.bboxes,
                tile_preds.labels,
                tile_preds.scores,
                saliency_maps,
                feature_vectors,
                strict=True,
            ):
                offset_x, offset_y, _, _ = tile_attr["roi"]
                tile_bboxes[:, 0::2] += offset_x
                tile_bboxes[:, 1::2] += offset_y

                tile_id = tile_attr["tile_id"]
                if tile_id not in img_ids:
                    img_ids.append(tile_id)
                tile_img_info.padding = tile_attr["roi"]

                det_pred_entity = DetPredEntity(
                    image=torch.empty(tile_img_info.ori_shape),
                    img_info=tile_img_info,
                    bboxes=tile_bboxes,
                    labels=tile_labels,
                    score=tile_scores,
                )

                if explain_mode:
                    det_pred_entity.feature_vector = tile_f_vect
                    det_pred_entity.saliency_map = tile_s_map
                entities_to_merge[tile_id].append(det_pred_entity)

        return [
            self._merge_entities(image_info, entities_to_merge[img_id], explain_mode)
            for img_id, image_info in zip(img_ids, self.img_infos, strict=True)
        ]

    def _merge_entities(
        self,
        img_info: ImageInfo,
        entities: list[DetPredEntity],
        explain_mode: bool = False,
    ) -> DetPredEntity:
        """Merge tile predictions to one single prediction.

        Args:
            img_info (ImageInfo): Image information about the original image before tiling.
            entities (list[DetPredEntity]): List of tile prediction entities.
            explain_mode (bool): Whether or not tiles have explain features. Default: False.

        Returns:
            DetPredEntity: Merged prediction entity.
        """
        bboxes: list | torch.Tensor = []
        labels: list | torch.Tensor = []
        scores: list | torch.Tensor = []
        feature_vectors = []
        saliency_maps = []
        tiles_coords = []
        img_size = img_info.ori_shape
        for tile_entity in entities:
            num_preds = len(tile_entity.bboxes)
            if num_preds > 0:
                bboxes.extend(tile_entity.bboxes)
                labels.extend(tile_entity.labels)
                scores.extend(tile_entity.score)
            if explain_mode:
                tiles_coords.append(tile_entity.img_info.padding)
                feature_vectors.append(tile_entity.feature_vector)
                saliency_maps.append(tile_entity.saliency_map)

        bboxes = torch.stack(bboxes) if len(bboxes) > 0 else torch.empty((0, 4), device=img_info.device)
        labels = torch.stack(labels) if len(labels) > 0 else torch.empty((0,), device=img_info.device)
        scores = torch.stack(scores) if len(scores) > 0 else torch.empty((0,), device=img_info.device)

        bboxes, labels, scores, _ = self.nms_postprocess(bboxes, scores, labels)

        det_pred_entity = DetPredEntity(
            image=torch.empty(img_size),
            img_info=img_info,
            score=scores,
            bboxes=tv_tensors.BoundingBoxes(bboxes, canvas_size=img_size, format="XYXY"),
            labels=labels,
        )

        if explain_mode:
            det_pred_entity.feature_vector = np.mean(feature_vectors, axis=0)
            det_pred_entity.saliency_map = self._merge_saliency_maps(saliency_maps, img_size, tiles_coords)

        return det_pred_entity

    def _merge_saliency_maps(
        self,
        saliency_maps: list[np.array],
        shape: tuple[int, int],
        tiles_coords: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Merging saliency maps from each tile for PyTorch implementation.

        OV implementation is on ModelAPI side.

        Args:
            saliency_maps: list of saliency maps, shape of each map is (Nc, H, W)
            shape: shape of the original image
            tiles_coords: coordinates of tiles

        Returns:
            Merged saliency map with shape (Nc, H, W)
        """
        if len(saliency_maps) == 1:
            return saliency_maps[0]

        image_saliency_map = saliency_maps[0]

        if len(image_saliency_map.shape) == 1:
            return image_saliency_map

        num_classes = saliency_maps[0].shape[0]
        map_h, map_w = saliency_maps[0].shape[1:]

        image_h, image_w = shape
        ratio = map_h / min(image_h, self.tile_size[0]), map_w / min(image_w, self.tile_size[1])

        image_map_h = int(image_h * ratio[0])
        image_map_w = int(image_w * ratio[1])
        merged_map = np.zeros((num_classes, image_map_h, image_map_w))

        # Note: Skip the first saliency map as it is the full image value.
        saliency_maps, start_idx = (saliency_maps[1:], 1) if self.with_full_img else (saliency_maps, 0)

        for i, saliency_map in enumerate(saliency_maps, start_idx):
            for class_idx in range(num_classes):
                cls_map = saliency_map[class_idx]

                x_1, y_1, map_w, map_h = tiles_coords[i]
                x_2, y_2 = x_1 + map_w, y_1 + map_h

                y_1, x_1 = int(y_1 * ratio[0]), int(x_1 * ratio[1])
                y_2, x_2 = int(y_2 * ratio[0]), int(x_2 * ratio[1])

                map_h, map_w = cls_map.shape

                if (map_h > y_2 - y_1 > 0) and (map_w > x_2 - x_1 > 0):
                    cls_map = cv2.resize(cls_map, (x_2 - x_1, y_2 - y_1))

                map_h, map_w = y_2 - y_1, x_2 - x_1

                for hi, wi in [(h_, w_) for h_ in range(map_h) for w_ in range(map_w)]:
                    map_pixel = cls_map[hi, wi]
                    merged_pixel = merged_map[class_idx][y_1 + hi, x_1 + wi]
                    if merged_pixel != 0:
                        merged_map[class_idx][y_1 + hi, x_1 + wi] = 0.5 * (map_pixel + merged_pixel)
                    else:
                        merged_map[class_idx][y_1 + hi, x_1 + wi] = map_pixel

        for class_idx in range(num_classes):
            if self.with_full_img:
                image_map_cls = image_saliency_map[class_idx]
                image_map_cls = cv2.resize(image_map_cls, (image_map_w, image_map_h))
                merged_map[class_idx] += 0.5 * image_map_cls

            merged_map[class_idx] = _non_linear_normalization(merged_map[class_idx])

        return merged_map.astype(np.uint8)


def _non_linear_normalization(saliency_map: np.ndarray) -> np.ndarray:
    """Use non-linear normalization y=x**1.5 for 2D saliency maps."""
    min_soft_score = np.min(saliency_map)
    # Make merged_map distribution positive to perform non-linear normalization y=x**1.5
    saliency_map = (saliency_map - min_soft_score) ** 1.5

    max_soft_score = np.max(saliency_map)
    saliency_map = 255.0 / (max_soft_score + 1e-12) * saliency_map

    return np.floor(saliency_map)


class InstanceSegTileMerge(TileMerge):
    """Instance segmentation tile merge."""

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
        explain_mode = self.explain_mode

        for tile_preds, tile_attrs in zip(batch_tile_preds, batch_tile_attrs, strict=True):
            feature_vectors = tile_preds.feature_vector if explain_mode else [[] for _ in range(len(tile_attrs))]
            for tile_attr, tile_img_info, tile_bboxes, tile_labels, tile_scores, tile_masks, tile_f_vect in zip(
                tile_attrs,
                tile_preds.imgs_info,
                tile_preds.bboxes,
                tile_preds.labels,
                tile_preds.scores,
                tile_preds.masks,
                feature_vectors,
                strict=True,
            ):
                keep_indices = tile_masks.to_sparse().sum((1, 2)).to_dense() > 0
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

                inst_seg_pred_entity = InstanceSegPredEntity(
                    image=torch.empty(tile_img_info.ori_shape),
                    img_info=tile_img_info,
                    bboxes=_bboxes,
                    labels=_labels,
                    score=_scores,
                    masks=_masks.to_sparse(),
                    polygons=[],
                )

                if explain_mode:
                    inst_seg_pred_entity.feature_vector = tile_f_vect
                    inst_seg_pred_entity.saliency_map = []
                entities_to_merge[tile_id].append(inst_seg_pred_entity)

        return [
            self._merge_entities(image_info, entities_to_merge[img_id], explain_mode)
            for img_id, image_info in zip(img_ids, self.img_infos, strict=True)
        ]

    def _merge_entities(
        self,
        img_info: ImageInfo,
        entities: list[InstanceSegPredEntity],
        explain_mode: bool = False,
    ) -> InstanceSegPredEntity:
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
        feature_vectors = []
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
            if explain_mode:
                feature_vectors.append(tile_entity.feature_vector)

        bboxes = torch.stack(bboxes) if len(bboxes) > 0 else torch.empty((0, 4), device=img_info.device)
        labels = torch.stack(labels) if len(labels) > 0 else torch.empty((0,), device=img_info.device)
        scores = torch.stack(scores) if len(scores) > 0 else torch.empty((0,), device=img_info.device)
        masks = masks if len(masks) > 0 else torch.empty((0, *img_size))

        bboxes, labels, scores, masks = self.nms_postprocess(bboxes, scores, labels, masks)

        inst_seg_pred_entity = InstanceSegPredEntity(
            image=torch.empty(img_size),
            img_info=img_info,
            score=scores,
            bboxes=tv_tensors.BoundingBoxes(bboxes, canvas_size=img_size, format="XYXY"),
            labels=labels,
            masks=tv_tensors.Mask(masks, dtype=bool),
            polygons=[],
        )

        if explain_mode:
            inst_seg_pred_entity.feature_vector = np.mean(feature_vectors, axis=0)
            inst_seg_pred_entity.saliency_map = self.get_saliency_maps_from_masks(
                labels,
                scores,
                masks,
                self.num_classes,
            )

        return inst_seg_pred_entity

    def get_saliency_maps_from_masks(
        self,
        labels: torch.Tensor,
        scores: torch.Tensor,
        masks: None | torch.Tensor,
        num_classes: int,
    ) -> np.ndarray:
        """Average and normalize predicted masks in  per-class.

        Returns:
            np.array: Class-wise Saliency Maps. One saliency map per each class - [class_id, H, W]
        """
        if masks is None:
            return np.ndarray([])

        pred = {"labels": labels, "scores": scores, "masks": masks}
        return InstSegExplainAlgo.average_and_normalize(pred, num_classes)
