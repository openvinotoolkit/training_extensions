"""Collection of video loading data pipelines for OTX Action Task."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Any, Dict

import numpy as np
from mmaction.datasets.builder import PIPELINES

from otx.api.entities.datasets import DatasetEntity


@PIPELINES.register_module(force=True)
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
