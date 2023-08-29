"""Collection of load pipelines for segmentation task."""
# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

from mmseg.datasets.builder import PIPELINES, build_from_cfg

import otx.algorithms.common.adapters.mmcv.pipelines.load_image_from_otx_dataset as load_image_base
from otx.algorithms.segmentation.adapters.mmseg.datasets.dataset import (
    get_annotation_mmseg_format,
)


# pylint: disable=too-many-instance-attributes, too-many-arguments
@PIPELINES.register_module()
class LoadImageFromOTXDataset(load_image_base.LoadImageFromOTXDataset):
    """Pipeline element that loads an image from a OTX Dataset on the fly."""

    def __init__(self, use_otx_adapter: bool = True, **kwargs):
        self.use_otx_adapter = use_otx_adapter
        super().__init__(**kwargs)


@PIPELINES.register_module()
class LoadResizeDataFromOTXDataset(load_image_base.LoadResizeDataFromOTXDataset):
    """Load and resize image & annotation with cache support."""

    def __init__(self, use_otx_adapter: bool = True, **kwargs):
        self.use_otx_adapter = use_otx_adapter
        super().__init__(**kwargs)

    def _create_load_ann_op(self, cfg: Optional[Dict]) -> Optional[Any]:
        """Creates resize operation."""
        if cfg is None:
            return None
        return build_from_cfg(cfg, PIPELINES)

    def _create_resize_op(self, cfg: Optional[Dict]) -> Optional[Any]:
        """Creates resize operation."""
        if cfg is None:
            return None
        return build_from_cfg(cfg, PIPELINES)


@PIPELINES.register_module()
class LoadAnnotationFromOTXDataset:
    """Pipeline element that loads an annotation from a OTX Dataset on the fly.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the annotation
        results['ann_info']['label_list']: list of all labels in the project

    """

    def __init__(self, use_otx_adapter=True):
        self.use_otx_adapter = use_otx_adapter

    def __call__(self, results: Dict[str, Any]):
        """Callback function of LoadAnnotationFromOTXDataset."""
        dataset_item = results.pop("dataset_item")  # Prevent unnessary deepcopy
        labels = results["ann_info"]["labels"]

        ann_info = get_annotation_mmseg_format(dataset_item, labels, self.use_otx_adapter)

        results["gt_semantic_seg"] = ann_info["gt_semantic_seg"]
        results["seg_fields"].append("gt_semantic_seg")

        return results
