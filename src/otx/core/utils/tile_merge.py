# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX tile merge module."""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import Generic

import cv2
import torch
from datumaro import Bbox, DatasetItem, Image, Mask
from datumaro import Dataset as DmDataset
from datumaro.plugins.tiling.merge_tile import MergeTile
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
        score_thres: float = 0.1,
        max_num_instances: int = 100,
    ) -> None:
        self.img_infos = img_infos
        self.score_thres = score_thres
        self.max_num_instances = max_num_instances

    def merge_dataset_items(self, dataset_items: list[DatasetItem]) -> DatasetItem:
        """Merge dataset items into one single dataset item.

        Args:
            dataset_items (list[DatasetItem]): List of tile dataset items.

        Returns:
            DatasetItem: Merged dataset item.
        """
        dataset = DmDataset.from_iterable(dataset_items)
        return dataset.transform(MergeTile).get(next(iter(dataset)).id)

    @abstractmethod
    def create_pred_entity(self, img_info: ImageInfo, merged_item: DatasetItem) -> T_OTXDataEntity:
        """Create merged prediction entity from merged dataset item.

        Args:
            img_info (ImageInfo): Image information.
            merged_item (DatasetItem): Merged dataset item.
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
        dataset_item_to_merge = defaultdict(list)
        anno_id = 0
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
                _tile_img = Image.from_numpy(cv2.resize(_tile_img.transpose(1, 2, 0), tile_img_info.ori_shape))
                _bboxes = tile_bboxes[keep_indices].detach().cpu().numpy()
                _labels = tile_labels[keep_indices].detach().cpu().numpy()
                _scores = tile_scores[keep_indices].detach().cpu().numpy()
                annotations = []

                for n in range(len(_bboxes)):
                    x1, y1, x2, y2 = _bboxes[n]
                    w, h = x2 - x1, y2 - y1
                    annotations.append(
                        Bbox(x1, y1, w, h, label=_labels[n], id=anno_id, attributes={"score": _scores[n]}),
                    )
                    anno_id += 1

                tile_idx = tile_attr["tile_idx"]
                tile_id = tile_attr["tile_id"]
                if tile_id not in img_ids:
                    img_ids.append(tile_id)
                dataset_item = DatasetItem(
                    media=_tile_img,
                    id=tile_idx,
                    annotations=annotations,
                    attributes=tile_attr,
                )

                dataset_item_to_merge[tile_id].append(dataset_item)

        predictions = []
        for img_id, image_info in zip(img_ids, self.img_infos):
            merged_item = self.merge_dataset_items(dataset_item_to_merge[img_id])
            predictions.append(self.create_pred_entity(image_info, merged_item))
        return predictions

    def create_pred_entity(self, img_info: ImageInfo, merged_item: DatasetItem) -> DetPredEntity:
        """Create merged detection prediction entity from merged datumaro dataset item.

        Args:
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

        pred_scores = torch.tensor(pred_scores, device=device)
        pred_bboxes = tv_tensors.BoundingBoxes(
            pred_bboxes,
            format="XYXY",
            canvas_size=img_info.ori_shape,
            device=device,
        )
        pred_labels = torch.tensor(pred_labels, device=device)
        sort_inds = torch.argsort(pred_scores, descending=True)
        if len(sort_inds) > self.max_num_instances:
            sort_inds = sort_inds[: self.max_num_instances]

        return DetPredEntity(
            image=tv_tensors.Image(merged_item.media_as(Image).data),
            img_info=img_info,
            score=pred_scores[sort_inds],
            bboxes=pred_bboxes[sort_inds],
            labels=pred_labels[sort_inds],
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
        dataset_item_to_merge = defaultdict(list)
        anno_id = 0
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
                _tile_img = Image.from_numpy(cv2.resize(_tile_img.transpose(1, 2, 0), tile_img_info.ori_shape))
                _bboxes = tile_bboxes[keep_indices].detach().cpu().numpy()
                _labels = tile_labels[keep_indices].detach().cpu().numpy()
                _scores = tile_scores[keep_indices].detach().cpu().numpy()
                _masks = tile_masks[keep_indices].detach().cpu().numpy()
                annotations = []

                for n in range(len(_bboxes)):
                    x1, y1, x2, y2 = _bboxes[n]
                    w, h = x2 - x1, y2 - y1
                    if _masks[n].sum() > 0:
                        annotations.extend(
                            [
                                Mask(_masks[n], label=_labels[n], id=anno_id, attributes={"score": _scores[n]}),
                                Bbox(x1, y1, w, h, label=_labels[n], id=anno_id, attributes={"score": _scores[n]}),
                            ],
                        )
                        anno_id += 1

                tile_idx = tile_attr["tile_idx"]
                tile_id = tile_attr["tile_id"]

                if tile_id not in img_ids:
                    img_ids.append(tile_id)
                dataset_item = DatasetItem(
                    media=_tile_img,
                    id=tile_idx,
                    annotations=annotations,
                    attributes=tile_attr,
                )

                dataset_item_to_merge[tile_id].append(dataset_item)

        predictions = []
        for img_id, image_info in zip(img_ids, self.img_infos):
            merged_item = self.merge_dataset_items(dataset_item_to_merge[img_id])
            predictions.append(self.create_pred_entity(image_info, merged_item))
        return predictions

    def create_pred_entity(self, img_info: ImageInfo, merged_item: DatasetItem) -> InstanceSegPredEntity:
        """Create merged inst-seg prediction entity from merged datumaro dataset item.

        Args:
            img_info (ImageInfo): Image information.
            merged_item (DatasetItem): Merged dataset item.

        Returns:
            DetPredEntity: Merged detection prediction.
        """
        device = img_info.device
        pred_bboxes, pred_labels, pred_scores, pred_masks = [], [], [], []
        pred_masks_by_label = {}
        for anno in merged_item.annotations:
            if isinstance(anno, Bbox) and anno.get_area() > 0:
                pred_bboxes.append(anno.points)
                pred_labels.append(anno.label)
                pred_scores.append(float(anno.attributes["score"]))
            if isinstance(anno, Mask):
                # NOTE: Datumaro tile merge does not give mask instances.
                # It merges mask instances to one mask.
                pred_masks_by_label[anno.label] = anno.image

        if len(pred_bboxes) == 0:
            pred_bboxes = torch.empty((0, 4))

        pred_scores = torch.tensor(pred_scores, device=device)
        pred_bboxes = tv_tensors.BoundingBoxes(
            pred_bboxes,
            format="XYXY",
            canvas_size=img_info.ori_shape,
            device=device,
        )
        pred_labels = torch.tensor(pred_labels, device=device)
        sort_inds = torch.argsort(pred_scores, descending=True)
        if len(sort_inds) > self.max_num_instances:
            sort_inds = sort_inds[: self.max_num_instances]

        pred_scores = pred_scores[sort_inds]
        pred_bboxes = pred_bboxes[sort_inds]
        pred_labels = pred_labels[sort_inds]

        for pred_box, pred_label in zip(pred_bboxes, pred_labels):
            x1, y1, x2, y2 = (int(value) for value in pred_box)
            pred_label_mask = pred_masks_by_label[int(pred_label)]
            # TODO (Eugene): Performance issue here if there are too many mask instances.
            # Ideally to optimize memory and speed we should convert it to RLE
            # or save the crop pred_mask[y1:y2, x1:x2]
            # https://github.com/openvinotoolkit/datumaro/pull/1194
            bitmask = torch.zeros(pred_label_mask.shape, dtype=bool, device=device)
            bitmask[y1:y2, x1:x2] = torch.tensor(pred_label_mask[y1:y2, x1:x2])
            pred_masks.append(bitmask)
        pred_masks = torch.stack(pred_masks) if len(pred_masks) > 0 else torch.empty((0, *img_info.ori_shape))

        return InstanceSegPredEntity(
            image=tv_tensors.Image(merged_item.media_as(Image).data),
            img_info=img_info,
            score=pred_scores,
            bboxes=pred_bboxes,
            labels=pred_labels,
            masks=pred_masks,
            polygons=[],
        )
