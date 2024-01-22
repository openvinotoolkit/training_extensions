# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX tile merge module."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pycocotools.mask as mask_utils
import torch
from datumaro import Bbox, DatasetItem, Mask
from datumaro import Dataset as DmDataset
from datumaro.plugins.tiling.merge_tile import MergeTile
from torchvision import tv_tensors

from otx.core.data.entity.detection import DetBatchPredEntity, DetPredEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity, InstanceSegPredEntity

if TYPE_CHECKING:
    import numpy as np

    from otx.core.data.entity.base import ImageInfo


class TileMerge:
    def __init__(
        self,
        img_infos: list[ImageInfo],
        score_thres: float = 0.1,
        max_num_instances: int = 500,
    ) -> None:
        """Initialize TileMerge.

        Args:
            img_infos (list[ImageInfo]): Original image information before tiling.
            score_thres (float, optional): Score threshold to filter out low score predictions. Defaults to 0.1.
        """
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

    def create_merged_detection_prediction(self, img_info: ImageInfo, merged_item: DatasetItem) -> DetPredEntity:
        """Create merged detection prediction.

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
        sort_inds = torch.argsort(pred_scores, descending=True)[: self.max_num_instances]

        return DetPredEntity(
            image=tv_tensors.Image(torch.empty(img_info.ori_shape)),
            img_info=img_info,
            score=pred_scores[sort_inds],
            bboxes=pred_bboxes[sort_inds],
            labels=pred_labels[sort_inds],
        )

    def create_merged_inst_seg_prediction(self, img_info: ImageInfo, merged_item: DatasetItem) -> InstanceSegPredEntity:
        """Create merged instance segmentation prediction.

        Args:
            img (np.ndarray): Original untiled image.
            img_info (ImageInfo): Image information.
            merged_item (DatasetItem): Merged dataset item.

        Returns:
            InstanceSegPredEntity: Merged inst-seg prediction.
        """
        device = img_info.device
        pred_bboxes, pred_labels, pred_scores, pred_mask_rles = [], [], [], []
        pred_masks_by_label = {}
        for anno in merged_item.annotations:
            if isinstance(anno, Bbox) and anno.get_area() > 100:
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
                bitmask = np.zeros(pred_label_mask.shape)
                bitmask[y1:y2, x1:x2] = torch.tensor(pred_label_mask[y1:y2, x1:x2])
                rle = mask_utils.encode(np.asfortranarray(pred_label_mask))
                pred_mask_rles.append(rle)

        return InstanceSegPredEntity(
            image=tv_tensors.Image(torch.empty(img_info.ori_shape)),
            img_info=img_info,
            score=torch.tensor(pred_scores),
            bboxes=tv_tensors.BoundingBoxes(pred_bboxes, format="XYXY", canvas_size=img_info.ori_shape),
            labels=torch.tensor(pred_labels),
            masks=[],
            polygons=[],
        )

    def merge(self, batch_tile_preds: list, batch_tile_attrs: list) -> list:
        dataset_item_to_merge = defaultdict(list)
        anno_id = 0
        img_ids = []

        for tile_preds, tile_attrs in zip(batch_tile_preds, batch_tile_attrs):
            for i, (tile_img, tile_img_info, tile_attr) in enumerate(
                zip(tile_preds.images, tile_preds.imgs_info, tile_attrs),
            ):
                tile_bboxes = tile_preds.bboxes[i].data.detach().cpu()
                tile_labels = tile_preds.labels[i].detach().cpu()
                tile_scores = tile_preds.scores[i].detach().cpu()

                keep_indices = tile_scores > self.score_thres
                keep_indices = keep_indices.nonzero(as_tuple=True)[0]
                _bboxes = tile_bboxes[keep_indices].numpy()
                _labels = tile_labels[keep_indices].numpy()
                _scores = tile_scores[keep_indices].numpy()
                if isinstance(tile_preds, InstanceSegBatchPredEntity):
                    _masks = tile_preds.masks[i][keep_indices].detach().cpu().numpy()
                annotations = []

                for n in range(len(_bboxes)):
                    x1, y1, x2, y2 = _bboxes[n]
                    w, h = x2 - x1, y2 - y1
                    if isinstance(tile_preds, InstanceSegBatchPredEntity):
                        if _masks[n].sum() > 10:
                            annotations.extend(
                                [
                                    Mask(_masks[n], label=_labels[n], id=anno_id, attributes={"score": _scores[n]}),
                                    Bbox(x1, y1, w, h, label=_labels[n], id=anno_id, attributes={"score": _scores[n]}),
                                ],
                            )
                    elif isinstance(tile_preds, DetBatchPredEntity):
                        annotations.append(
                            Bbox(x1, y1, w, h, label=_labels[n], id=anno_id, attributes={"score": _scores[n]}),
                        )
                    anno_id += 1

                tile_idx = tile_attr["tile_idx"]
                tile_id = tile_attr["tile_id"]

                if tile_id not in img_ids:
                    img_ids.append(tile_id)
                dataset_item = DatasetItem(
                    id=tile_idx,
                    annotations=annotations,
                    attributes=tile_attr,
                )

                dataset_item_to_merge[tile_id].append(dataset_item)

        if len(img_ids) != len(self.img_infos):
            msg = (f"Number of image ids {len(img_ids)} does not match number of image infos {len(img_infos)}",)
            raise ValueError(msg)

        predictions = []
        for img_id, image_info in zip(img_ids, self.img_infos):
            merged_item = self.merge_dataset_items(dataset_item_to_merge[img_id])
            if isinstance(batch_tile_preds[0], InstanceSegBatchPredEntity):
                pred = self.create_merged_inst_seg_prediction(image_info, merged_item)
            else:
                pred = self.create_merged_detection_prediction(image_info, merged_item)
            predictions.append(pred)
        return predictions
