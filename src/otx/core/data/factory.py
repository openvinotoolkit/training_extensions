# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Factory classes for dataset and transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.core.types.task import OTXTaskType
from otx.core.types.transformer_libs import TransformLibType

from .dataset.base import OTXDataset, Transforms

if TYPE_CHECKING:
    from datumaro import DatasetSubset

    from otx.core.config.data import SubsetConfig

__all__ = ["TransformLibFactory", "OTXDatasetFactory"]


class TransformLibFactory:
    """Factory class for transform."""

    @classmethod
    def generate(cls: type[TransformLibFactory], config: SubsetConfig) -> Transforms:
        """Create transforms from factory."""
        if config.transform_lib_type == TransformLibType.MMCV:
            from .transform_libs.mmcv import MMCVTransformLib

            return MMCVTransformLib.generate(config)
        
        if config.transform_lib_type == TransformLibType.MMPRETRAIN:
            from .transform_libs.mmpretrain import MMPretrainTransformLib

            return MMPretrainTransformLib.generate(config)

        if config.transform_lib_type == TransformLibType.MMDET:
            from .transform_libs.mmdet import MMDetTransformLib

            return MMDetTransformLib.generate(config)

        raise NotImplementedError(config.transform_lib_type)


class OTXDatasetFactory:
    """Factory class for OTXDataset."""

    @classmethod
    def create(
        cls: type[OTXDatasetFactory],
        task: OTXTaskType,
        dm_subset: DatasetSubset,
        config: SubsetConfig,
    ) -> OTXDataset:
        """Create OTXDataset."""
        transforms = TransformLibFactory.generate(config)
        if task == OTXTaskType.MULTI_CLASS_CLS:
            from .dataset.classification import OTXMulticlassClsDataset

            return OTXMulticlassClsDataset(dm_subset, transforms)
        
        if task == OTXTaskType.DETECTION:
            from .dataset.detection import OTXDetectionDataset

            return OTXDetectionDataset(dm_subset, transforms)

        raise NotImplementedError(task)
