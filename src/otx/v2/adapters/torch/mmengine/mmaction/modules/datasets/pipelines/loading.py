"""Collection of video loading data pipelines for OTX Action Task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import numpy as np
from mmaction.registry import TRANSFORMS

from otx.api.entities.datasets import DatasetEntity


@TRANSFORMS.register_module(force=True)
class RawFrameDecode:
    """Load and decode frames with given indices."""

    otx_dataset: DatasetEntity

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Call function of RawFrameDecode."""
        results = self._decode_from_list(results)
        return results

    def _decode_from_list(self, results: Dict[str, Any]):
        """Generate numpy array list from list of DatasetItemEntity."""
        imgs = []
        for index in results["frame_inds"]:
            imgs.append(self.otx_dataset[int(index)].media.numpy)
        results["imgs"] = imgs
        results["original_shape"] = imgs[0].shape[:2]
        results["img_shape"] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if "gt_bboxes" in results:
            height, width = results["img_shape"]
            scale_factor = np.array([width, height, width, height])
            gt_bboxes = results["gt_bboxes"]
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results["gt_bboxes"] = gt_bboxes
            if "proposals" in results and results["proposals"] is not None:
                proposals = results["proposals"]
                proposals = (proposals * scale_factor).astype(np.float32)
                results["proposals"] = proposals

        return results
