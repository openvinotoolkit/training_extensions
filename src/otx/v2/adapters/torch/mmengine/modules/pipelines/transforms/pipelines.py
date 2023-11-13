"""Collection Pipeline for mmengine task."""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from mmengine.registry import TRANSFORMS

if TYPE_CHECKING:
    from mmcv.transforms import BaseTransform

import otx.v2.adapters.datumaro.pipelines.load_image_from_otx_dataset as load_image_base


@TRANSFORMS.register_module()
class LoadImageFromOTXDataset(load_image_base.LoadImageFromOTXDataset):
    """Pipeline element that loads an image from a OTX Dataset on the fly."""


@TRANSFORMS.register_module()
class LoadResizeDataFromOTXDataset(load_image_base.LoadResizeDataFromOTXDataset):
    """Load and resize image & annotation with cache support."""

    def _create_resize_op(self, cfg: dict | None = None) -> BaseTransform | None:
        """Creates resize operation."""
        return TRANSFORMS.build(cfg)
