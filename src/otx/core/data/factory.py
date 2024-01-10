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

    from otx.core.config.data import DataModuleConfig, InstSegDataModuleConfig, SubsetConfig
    from otx.core.data.mem_cache import MemCacheHandlerBase


__all__ = ["TransformLibFactory", "OTXDatasetFactory"]


class TransformLibFactory:
    """Factory class for transform."""

    @classmethod
    def generate(cls: type[TransformLibFactory], config: SubsetConfig) -> Transforms:
        """Create transforms from factory."""
        if config.transform_lib_type == TransformLibType.TORCHVISION:
            from .transform_libs.torchvision import TorchVisionTransformLib

            return TorchVisionTransformLib.generate(config)

        if config.transform_lib_type == TransformLibType.MMCV:
            from .transform_libs.mmcv import MMCVTransformLib

            return MMCVTransformLib.generate(config)

        if config.transform_lib_type == TransformLibType.MMPRETRAIN:
            from .transform_libs.mmpretrain import MMPretrainTransformLib

            return MMPretrainTransformLib.generate(config)

        if config.transform_lib_type == TransformLibType.MMDET:
            from .transform_libs.mmdet import MMDetTransformLib

            return MMDetTransformLib.generate(config)

        if config.transform_lib_type == TransformLibType.MMSEG:
            from .transform_libs.mmseg import MMSegTransformLib

            return MMSegTransformLib.generate(config)

        if config.transform_lib_type == TransformLibType.MMACTION:
            from .transform_libs.mmaction import MMActionTransformLib

            return MMActionTransformLib.generate(config)

        raise NotImplementedError(config.transform_lib_type)


class OTXDatasetFactory:
    """Factory class for OTXDataset."""

    @classmethod
    def create(
        cls: type[OTXDatasetFactory],
        task: OTXTaskType,
        dm_subset: DatasetSubset,
        mem_cache_handler: MemCacheHandlerBase,
        cfg_subset: SubsetConfig,
        cfg_data_module: DataModuleConfig | InstSegDataModuleConfig,
    ) -> OTXDataset:
        """Create OTXDataset."""
        transforms = TransformLibFactory.generate(cfg_subset)
        if task == OTXTaskType.MULTI_CLASS_CLS:
            from .dataset.classification import OTXMulticlassClsDataset

            return OTXMulticlassClsDataset(
                dm_subset=dm_subset,
                transforms=transforms,
                mem_cache_handler=mem_cache_handler,
                mem_cache_img_max_size=cfg_data_module.mem_cache_img_max_size,
            )

        if task == OTXTaskType.MULTI_LABEL_CLS:
            from .dataset.classification import OTXMultilabelClsDataset

            return OTXMultilabelClsDataset(
                dm_subset=dm_subset,
                transforms=transforms,
                mem_cache_handler=mem_cache_handler,
                mem_cache_img_max_size=cfg_data_module.mem_cache_img_max_size,
            )

        if task == OTXTaskType.DETECTION:
            from .dataset.detection import OTXDetectionDataset

            return OTXDetectionDataset(
                dm_subset=dm_subset,
                transforms=transforms,
                mem_cache_handler=mem_cache_handler,
                mem_cache_img_max_size=cfg_data_module.mem_cache_img_max_size,
            )

        if task == OTXTaskType.INSTANCE_SEGMENTATION:
            from .dataset.instance_segmentation import OTXInstanceSegDataset

            # NOTE: DataModuleConfig does not have include_polygons attribute
            include_polygons = getattr(cfg_data_module, "include_polygons", False)
            return OTXInstanceSegDataset(
                dm_subset=dm_subset,
                transforms=transforms,
                mem_cache_handler=mem_cache_handler,
                mem_cache_img_max_size=cfg_data_module.mem_cache_img_max_size,
                include_polygons=include_polygons,
            )

        if task == OTXTaskType.SEMANTIC_SEGMENTATION:
            from .dataset.segmentation import OTXSegmentationDataset

            return OTXSegmentationDataset(
                dm_subset=dm_subset,
                transforms=transforms,
                mem_cache_handler=mem_cache_handler,
                mem_cache_img_max_size=cfg_data_module.mem_cache_img_max_size,
            )

        if task == OTXTaskType.ACTION_CLASSIFICATION:
            from .dataset.action_classification import OTXActionClsDataset

            return OTXActionClsDataset(
                dm_subset=dm_subset,
                transforms=transforms,
                mem_cache_handler=mem_cache_handler,
                mem_cache_img_max_size=cfg_data_module.mem_cache_img_max_size,
            )

        if task == OTXTaskType.ACTION_DETECTION:
            from .dataset.action_detection import OTXActionDetDataset

            return OTXActionDetDataset(
                dm_subset=dm_subset,
                transforms=transforms,
                mem_cache_handler=mem_cache_handler,
                mem_cache_img_max_size=cfg_data_module.mem_cache_img_max_size,
            )
            
        if task == OTXTaskType.VISUAL_PROMPTING:
            from .dataset.visual_prompting import OTXVisualPromptingDataset

            # NOTE: DataModuleConfig does not have include_polygons attribute
            include_polygons = getattr(cfg_data_module, "include_polygons", False)
            return OTXVisualPromptingDataset(
                dm_subset=dm_subset,
                transforms=transforms,
                mem_cache_handler=mem_cache_handler,
                mem_cache_img_max_size=cfg_data_module.mem_cache_img_max_size,
                include_polygons=include_polygons,
            )
        raise NotImplementedError(task)
