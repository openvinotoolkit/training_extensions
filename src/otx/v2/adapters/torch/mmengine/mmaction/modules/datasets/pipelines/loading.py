"""Collection of video loading data pipelines for OTX Action Task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mmaction.registry import TRANSFORMS

if TYPE_CHECKING:
    from datumaro.components.dataset import Dataset as DatumDataset

@TRANSFORMS.register_module(force=True)
class OTXRawFrameDecode:
    """Load and decode frames with given indices."""

    otx_dataset: DatumDataset

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        """Call function of RawFrameDecode."""
        return self._decode_from_list(results)

    def _decode_from_list(self, results: dict[str, Any]) -> dict:
        """Generate numpy array list from list of DatasetItemEntity."""
        imgs = []
        for index in results["frame_inds"]:
            item_id = results["item_ids"][index]
            imgs.append(self.otx_dataset.get(id=item_id, subset="annotations").media.data)

        results["imgs"] = imgs
        results["original_shape"] = imgs[0].shape[:2]
        results["img_shape"] = imgs[0].shape[:2]

        return results
