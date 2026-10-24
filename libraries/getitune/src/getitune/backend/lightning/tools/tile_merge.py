# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""getitune tile merge module."""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np
import torch
from packaging import version
from torchvision import tv_tensors
from torchvision.ops import batched_nms

from getitune.backend.lightning.tools.explain.explain_algo import InstSegExplainAlgo
from getitune.config.data import TileConfig
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import Prediction, PredictionBatch

if TYPE_CHECKING:
    from datumaro.experimental.fields import TileInfo

# Maximum number of elements 2**31 -1
MAX_ELEMENTS: int = np.iinfo(np.int32).max


# NOTE: RuntimeError: nonzero is not supported for tensors with more than INT_MAX elements,
# See https://github.com/pytorch/pytorch/issues/51871
int_max_check_condition: Callable[[torch.Tensor], bool] = (
    lambda tile_masks: version.parse(torch.__version__) < version.parse("2.6")
    and torch.numel(tile_masks) > MAX_ELEMENTS
)


def keep_chunkify(tensor: torch.Tensor, max_element: int = MAX_ELEMENTS) -> torch.Tensor:
    """Splits tensor into chunks and processes each chunk separately.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, H, W).

    Returns:
        torch.Tensor: Boolean mask of shape (B,) indicating nonzero sum.
    """
    _, h, w = tensor.shape
    max_batch_size = int(max_element) // (h * w)
    chunk_size = max(1, min(max_batch_size, tensor.shape[0]))

    keep_indices = []
    for i in range(0, tensor.shape[0], chunk_size):
        chunk = tensor[i : i + chunk_size]
        keep_indices.append(chunk.sum(dim=(1, 2)) > 0)  # Process chunk

    return torch.cat(keep_indices, dim=0)


class TileMerge:
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
        self.explain_mode = explain_mode

    @abstractmethod
    def _merge_entities(
        self,
        img_info: ImageInfo,
        entities: list[Prediction],
        explain_mode: bool = False,
    ) -> Prediction:
        """Merge tile predictions to one single full-size prediction data entity.

        Args:
            img_info (ImageInfo): Image information about the original image before tiling.
            entities (list[Prediction]): List of tile prediction entities.
            explain_mode (bool): Whether or not tiles have explain features. Default: False.

        Returns:
            Prediction: Merged prediction entity.
        """
        raise NotImplementedError

    @abstractmethod
    def merge(
        self,
        batch_tile_preds: list[PredictionBatch],
        batch_tile_infos: list[list[dict]],
    ) -> list[Prediction]:
        """Merge batch tile predictions to a list of full-size prediction data entities.

        Args:
            batch_tile_preds (list): list of tile predictions.
            batch_tile_infos (list): list of tile attributes.
        """
        raise NotImplementedError

    def nms_postprocess(
        self,
        bboxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        masks: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        batch_tile_preds: list[PredictionBatch],
        batch_tile_infos: list[list[TileInfo]],
    ) -> list[Prediction]:
        """Merge batch tile predictions to a list of full-size prediction data entities.

        Args:
            batch_tile_preds (list): detection tile predictions.
            batch_tile_infos (list): detection tile attributes.

        """
        entities_to_merge = defaultdict(list)
        img_ids = []
        explain_mode = self.explain_mode

        for tile_preds, tile_infos in zip(batch_tile_preds, batch_tile_infos, strict=True):
            if tile_preds.imgs_info is None or tile_preds.bboxes is None:
                msg = "imgs_info or bboxes is None"
                raise ValueError(msg)
            batch_size = len(tile_infos)
            for i in range(batch_size):
                if tile_preds.imgs_info[i] is None:
                    msg = "imgs_info is None"
                    raise ValueError(msg)
                tile_img_info = tile_preds.imgs_info[i]
                tile_info = tile_infos[i]
                tile_s_map = tile_preds.saliency_map[i] if tile_preds.saliency_map is not None else None
                tile_f_vect = tile_preds.feature_vector[i] if tile_preds.feature_vector is not None else None

                tile_bboxes = tile_preds.bboxes[i] if tile_preds.bboxes[i].numel() > 0 else None
                offset_x = tile_info.x
                offset_y = tile_info.y
                if tile_bboxes is not None:
                    tile_bboxes[:, 0::2] += offset_x
                    tile_bboxes[:, 1::2] += offset_y

                tile_id = tile_info.source_sample_idx
                if tile_id not in img_ids:
                    img_ids.append(tile_id)
                tile_img_info.padding = [tile_info.x, tile_info.y, tile_info.width, tile_info.height]  # type: ignore[union-attr]

                det_pred_entity = Prediction(
                    image=torch.empty(3, *tile_img_info.ori_shape),  # type: ignore[union-attr]
                    img_info=tile_img_info,
                    bboxes=tile_bboxes,
                    label=tile_preds.labels[i] if tile_preds.labels is not None else None,
                    scores=tile_preds.scores[i] if tile_preds.scores is not None else None,
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
        entities: list[Prediction],
        explain_mode: bool = False,
    ) -> Prediction:
        """Merge tile predictions to one single prediction.

        Args:
            img_info (ImageInfo): Image information about the original image before tiling.
            entities (list[DetPredEntity]): List of tile prediction entities.
            explain_mode (bool): Whether or not tiles have explain features. Default: False.

        Returns:
            TorchPredItem: Merged prediction entity.
        """
        bboxes: list | torch.Tensor = []
        labels: list | torch.Tensor = []
        scores: list | torch.Tensor = []
        feature_vectors = []
        saliency_maps = []
        tiles_coords = []
        img_size = img_info.ori_shape
        for tile_entity in entities:
            num_preds = len(tile_entity.bboxes) if tile_entity.bboxes is not None else 0
            if num_preds > 0:
                bboxes.extend(tile_entity.bboxes if tile_entity.bboxes is not None else [])
                labels.extend(tile_entity.label if tile_entity.label is not None else [])
                scores.extend(tile_entity.scores if tile_entity.scores is not None else [])
            if explain_mode:
                tiles_coords.append(tile_entity.img_info.padding)  # type: ignore[union-attr]
                if tile_entity.feature_vector is not None:
                    feature_vectors.append(tile_entity.feature_vector.cpu().numpy())
                if tile_entity.saliency_map is not None:
                    saliency_maps.append(tile_entity.saliency_map.cpu().numpy())

        bboxes = torch.stack(bboxes) if len(bboxes) > 0 else torch.empty((0, 4), device=img_info.device)
        labels = torch.stack(labels) if len(labels) > 0 else torch.empty((0,), dtype=torch.long, device=img_info.device)
        scores = torch.stack(scores) if len(scores) > 0 else torch.empty((0,), device=img_info.device)

        bboxes, labels, scores, _ = self.nms_postprocess(bboxes, scores, labels)

        det_pred_entity = Prediction(
            image=torch.empty(3, *img_size),
            img_info=img_info,
            scores=scores,
            bboxes=tv_tensors.BoundingBoxes(bboxes, canvas_size=img_size, format="XYXY"),
            label=labels,
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

        for i, saliency_map in enumerate(saliency_maps):
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
        self, batch_tile_preds: list[PredictionBatch], batch_tile_infos: list[list[TileInfo]]
    ) -> list[Prediction]:
        """Merge inst-seg tile predictions to one single prediction.

        Args:
            batch_tile_preds (list): instance-seg tile predictions.

        """
        entities_to_merge = defaultdict(list)
        img_ids = []
        explain_mode = self.explain_mode

        for tile_preds, tile_infos in zip(batch_tile_preds, batch_tile_infos, strict=True):
            feature_vectors = tile_preds.feature_vector if explain_mode else [[] for _ in range(len(tile_infos))]
            for i in range(len(tile_infos)):
                tile_info = tile_infos[i]
                tile_img_info = tile_preds.imgs_info[i] if tile_preds.imgs_info is not None else None
                tile_bboxes = tile_preds.bboxes[i] if tile_preds.bboxes is not None else None
                tile_labels = tile_preds.labels[i] if tile_preds.labels is not None else None
                tile_scores = tile_preds.scores[i] if tile_preds.scores is not None else None
                tile_masks = tile_preds.masks[i] if tile_preds.masks is not None else None
                tile_f_vect = feature_vectors[i] if feature_vectors is not None else None

                if int_max_check_condition(tile_masks):
                    keep_indices = keep_chunkify(tile_masks)
                else:
                    keep_indices = tile_masks.to_sparse().sum((1, 2)).to_dense() > 0
                keep_indices = keep_indices.nonzero(as_tuple=True)[0]
                _bboxes = tile_bboxes[keep_indices]
                _labels = tile_labels[keep_indices]
                _scores = tile_scores[keep_indices]
                _masks = tile_masks[keep_indices]

                offset_x = tile_info.x
                offset_y = tile_info.y
                _bboxes[:, 0::2] += offset_x
                _bboxes[:, 1::2] += offset_y

                tile_id = tile_info.source_sample_idx
                if tile_id not in img_ids:
                    img_ids.append(tile_id)
                tile_img_info.padding = [tile_info.x, tile_info.y, tile_info.width, tile_info.height]  # type: ignore[union-attr]

                inst_seg_pred_entity = Prediction(
                    image=torch.empty(3, *tile_img_info.ori_shape),  # type: ignore[union-attr]
                    img_info=tile_img_info,
                    bboxes=tv_tensors.BoundingBoxes(_bboxes, canvas_size=tile_img_info.ori_shape, format="XYXY"),  # type: ignore[union-attr]
                    label=_labels,
                    scores=_scores,
                    masks=tv_tensors.Mask(_masks),
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
        entities: list[Prediction],
        explain_mode: bool = False,
    ) -> Prediction:
        """Merge tile predictions to one single prediction.

        Args:
            img_info (ImageInfo): Image information about the original image before tiling.
            entities (list[TorchPredItem]): List of tile prediction entities.

        Returns:
            TorchPredItem: Merged prediction entity.
        """
        bboxes: list | torch.Tensor = []
        labels: list | torch.Tensor = []
        scores: list | torch.Tensor = []
        masks: list | torch.Tensor = []
        feature_vectors = []
        img_size = img_info.ori_shape
        for tile_entity in entities:
            num_preds = len(tile_entity.bboxes) if tile_entity.bboxes is not None else 0
            if num_preds > 0:
                bboxes.extend(tile_entity.bboxes if tile_entity.bboxes is not None else [])
                labels.extend(tile_entity.label if tile_entity.label is not None else [])
                scores.extend(tile_entity.scores if tile_entity.scores is not None else [])

                offset_x, offset_y, _, _ = tile_entity.img_info.padding  # type: ignore[union-attr]
                mask_indices = tile_entity.masks.to_sparse().indices() if tile_entity.masks is not None else None
                mask_values = tile_entity.masks.to_sparse().values() if tile_entity.masks is not None else None
                if mask_indices is not None and mask_values is not None:
                    mask_indices[1] += offset_y
                    mask_indices[2] += offset_x
                    masks.extend(
                        torch.sparse_coo_tensor(mask_indices, mask_values, (num_preds, *img_size)),
                    )
            if explain_mode:
                feature_vectors.append(tile_entity.feature_vector)

        bboxes = torch.stack(bboxes) if len(bboxes) > 0 else torch.empty((0, 4), device=img_info.device)
        labels = torch.stack(labels) if len(labels) > 0 else torch.empty((0,), dtype=torch.long, device=img_info.device)
        scores = torch.stack(scores) if len(scores) > 0 else torch.empty((0,), device=img_info.device)
        masks = masks if len(masks) > 0 else torch.empty((0, *img_size))

        bboxes, labels, scores, masks = self.nms_postprocess(bboxes, scores, labels, masks)

        inst_seg_pred_entity = Prediction(
            image=torch.empty(3, *img_size),
            img_info=img_info,
            scores=scores,
            bboxes=tv_tensors.BoundingBoxes(bboxes, canvas_size=img_size, format="XYXY"),
            label=labels,
            masks=tv_tensors.Mask(masks, dtype=bool),
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
        masks: torch.Tensor | None,
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


class SegmentationTileMerge(TileMerge):
    """Semantic segmentation tile merge."""

    def __init__(
        self,
        img_infos: list[ImageInfo],
        num_classes: int,
        tile_config: TileConfig,
        explain_mode: bool = False,
    ) -> None:
        super().__init__(img_infos, num_classes, tile_config, explain_mode)
        if explain_mode:
            msg = "Explain mode is not supported for segmentation"
            raise ValueError(msg)

    def merge(
        self,
        batch_tile_preds: list[PredictionBatch],
        batch_tile_infos: list[list[TileInfo]],
    ) -> list[Prediction]:
        """Merge batch tile predictions to a list of full-size prediction data entities.

        Args:
            batch_tile_preds (list[SegBatchPredEntity]): segmentation tile predictions.
            batch_tile_infos (list[list[dict]]): segmentation tile attributes.

        Returns:
            list[TorchPredItem]: List of full-size prediction data entities after merging.
        """
        entities_to_merge = defaultdict(list)
        img_ids = []
        explain_mode = self.explain_mode

        for tile_preds, tile_infos in zip(batch_tile_preds, batch_tile_infos):
            batch_size = tile_preds.batch_size
            saliency_maps = tile_preds.saliency_map if explain_mode else [[] for _ in range(batch_size)]
            feature_vectors = tile_preds.feature_vector if explain_mode else [[] for _ in range(batch_size)]
            if saliency_maps is None or feature_vectors is None:
                msg = "The saliency maps or feature vectors are not provided."
                raise ValueError(msg)
            if tile_preds.imgs_info is None:
                msg = "Image information is not provided."
                raise ValueError(msg)
            if tile_preds.masks is None:
                msg = "The predicted masks are not provided."
                raise ValueError(msg)

            for tile_info, tile_img_info, tile_masks, tile_s_map, tile_f_vect in zip(
                tile_infos,
                tile_preds.imgs_info,
                tile_preds.masks,
                saliency_maps,
                feature_vectors,
            ):
                if tile_img_info is None:
                    msg = f"Image information is not provided : {tile_preds.imgs_info}."
                    raise ValueError(msg)

                tile_id = tile_info.source_sample_idx
                if tile_id not in img_ids:
                    img_ids.append(tile_id)
                tile_img_info.padding = (tile_info.x, tile_info.y, tile_info.width, tile_info.height)
                seg_pred_entity = Prediction(
                    image=torch.empty((3, *tile_img_info.ori_shape)),
                    img_info=tile_img_info,
                    masks=tv_tensors.Mask(tile_masks),
                    scores=torch.tensor([]),
                )

                if explain_mode:
                    seg_pred_entity.feature_vector = tile_f_vect
                    seg_pred_entity.saliency_map = tile_s_map
                entities_to_merge[tile_id].append(seg_pred_entity)

        return [
            self._merge_entities(image_info, entities_to_merge[img_id], explain_mode)
            for img_id, image_info in zip(img_ids, self.img_infos)
        ]

    def _merge_entities(
        self,
        img_info: ImageInfo,
        entities: list[Prediction],
        explain_mode: bool = False,
    ) -> Prediction:
        """Merge tile predictions to one single prediction.

        Args:
            img_info (ImageInfo): Image information about the original image before tiling.
            entities (list[TorchPredItem]): List of tile prediction entities.
            explain_mode (bool): Whether or not tiles have explain features. Default: False.

        Returns:
            TorchPredItem: Merged prediction entity.
        """
        img_size = img_info.ori_shape
        if any(entity is None for entity in entities):
            msg = f"Some entities are None: {entities}."
            raise ValueError(msg)
        if entities[0].masks is None:
            msg = "The predicted masks are not provided."
            raise ValueError(msg)
        num_classes = len(entities[0].masks)

        # Create a vote map for overlapping tiles
        vote_mask = torch.zeros(img_size, dtype=torch.int, device=img_info.device)
        full_logits_mask = torch.zeros((num_classes, *img_size), device=img_info.device)

        for tile_entity in entities:
            if tile_entity.img_info is None:
                msg = "Image information is not provided."
                raise ValueError(msg)
            if tile_entity.masks is None:
                msg = "The predicted masks are not provided."
                raise ValueError(msg)
            offset_x, offset_y, tile_w, tile_h = tile_entity.img_info.padding
            vote_mask[offset_y : offset_y + tile_h, offset_x : offset_x + tile_w] += 1
            full_logits_mask[:, offset_y : offset_y + tile_h, offset_x : offset_x + tile_w] += tile_entity.masks[
                :,
                :tile_h,
                :tile_w,
            ]
        full_logits_mask = full_logits_mask / vote_mask.unsqueeze(0)

        return Prediction(
            image=torch.empty((3, *img_size)),
            img_info=img_info,
            masks=tv_tensors.Mask(full_logits_mask.argmax(0).unsqueeze(0)),
            scores=torch.tensor([]),
        )
