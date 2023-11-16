"""Data Pipeline for MMSegmentation Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from otx.v2.adapters.torch.mmengine.mmseg.registry import TRANSFORMS

if TYPE_CHECKING:
    from datumaro.components.dataset_base import DatasetItem as DatumDatasetItem


@TRANSFORMS.register_module()
class LoadAnnotationFromOTXDataset:
    """Pipeline element that loads an annotation from a OTX Dataset on the fly."""

    def _load_seg_map(self, dataset_item: DatumDatasetItem) -> np.ndarray:
        """Load segmentation map from Datumaro DatasetItem.

        Args:
            dataset_item (DatumDatasetItem): The Datumaro DatasetItem to load the segmentation map from.

        Returns:
            np.ndarray: The segmentation map.
        """
        height, width = dataset_item.image.size
        mask = np.zeros(shape=(height, width), dtype=np.uint8)
        for annotation in dataset_item.annotations:
            bool_map: np.ndarray = annotation.image == 1
            mask[bool_map] = annotation.label + 1
        return mask

    def __call__(self, results: dict[str, Any]) -> dict:
        """Callback function of LoadAnnotationFromOTXDataset."""
        dataset_item = results.pop("dataset_item")
        results["gt_seg_map"] = self._load_seg_map(dataset_item)
        results["seg_fields"].append("gt_seg_map")

        return results
