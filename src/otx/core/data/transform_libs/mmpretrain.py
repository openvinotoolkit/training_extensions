# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Helper to support MMPretrain data transform functions."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

from mmpretrain.datasets.transforms import (
    PackInputs as MMPretrainPackInputs,
)
from mmpretrain.registry import TRANSFORMS
from torchvision import tv_tensors

from otx.core.data.entity.classification import HlabelClsDataEntity, MulticlassClsDataEntity, MultilabelClsDataEntity

from .mmcv import LoadImageFromFile, MMCVTransformLib

TRANSFORMS.register_module(module=LoadImageFromFile, force=True)

if TYPE_CHECKING:
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig


@TRANSFORMS.register_module(force=True)
class PackInputs(MMPretrainPackInputs):
    """Class to override PackInputs."""

    def transform(self, results: dict) -> MulticlassClsDataEntity | MultilabelClsDataEntity | HlabelClsDataEntity:
        """Pack MMPretrain data entity into MulticlassClsDataEntity."""
        otx_data_entity = results["__otx__"]

        if isinstance(otx_data_entity, MulticlassClsDataEntity):
            return self._pack_multiclass_inputs(results)
        if isinstance(otx_data_entity, MultilabelClsDataEntity):
            return self._pack_multilabel_inputs(results)
        if isinstance(otx_data_entity, HlabelClsDataEntity):
            return self._pack_hlabel_inputs(results)

        msg = "Unsupported data entity type"
        raise TypeError(msg)

    def _pack_common_inputs(self, results: dict) -> dict[str, Any]:
        transformed = super().transform(results)
        image = tv_tensors.Image(transformed.get("inputs"))
        data_samples = transformed["data_samples"]

        # Some MM* transforms return (H, W, C), not (H, W)
        img_shape = data_samples.img_shape if len(data_samples.img_shape) == 2 else data_samples.img_shape[:2]
        ori_shape = data_samples.ori_shape
        scale_factor = data_samples.metainfo.get("scale_factor", (1.0, 1.0))  # (W, H) because it is from mm pipeline

        data_entity: MulticlassClsDataEntity | MultilabelClsDataEntity | HlabelClsDataEntity = results["__otx__"]

        image_info = deepcopy(data_entity.img_info)
        image_info.img_shape = img_shape
        image_info.ori_shape = ori_shape
        image_info.scale_factor = scale_factor[::-1]  # convert to (H, W)

        return {
            "image": image,
            "image_info": image_info,
            "labels": results["__otx__"].labels,
        }

    def _pack_multiclass_inputs(self, results: dict) -> MulticlassClsDataEntity:
        """Pack multiclass classification inputs."""
        packed_common_inputs = self._pack_common_inputs(results)

        return MulticlassClsDataEntity(
            image=packed_common_inputs["image"],
            img_info=packed_common_inputs["image_info"],
            labels=packed_common_inputs["labels"],
        )

    def _pack_multilabel_inputs(self, results: dict) -> MultilabelClsDataEntity:
        """Pack multilabel classification inputs.

        NOTE,
        Currently, this function is the same with multiclass case.
        However, it should have different functionality if we consider the ignore_label.
        That's the reason why I reamined same function.
        """
        packed_common_inputs = self._pack_common_inputs(results)

        return MultilabelClsDataEntity(
            image=packed_common_inputs["image"],
            img_info=packed_common_inputs["image_info"],
            labels=packed_common_inputs["labels"],
        )

    def _pack_hlabel_inputs(self, results: dict) -> HlabelClsDataEntity:
        """Pack hlabel classification inputs."""
        packed_common_inputs = self._pack_common_inputs(results)

        return HlabelClsDataEntity(
            image=packed_common_inputs["image"],
            img_info=packed_common_inputs["image_info"],
            labels=packed_common_inputs["labels"],
        )


class MMPretrainTransformLib(MMCVTransformLib):
    """Helper to support MMPretrain transforms in OTX."""

    @classmethod
    def get_builder(cls) -> Registry:
        """Transform builder obtained from MMPretrain."""
        return TRANSFORMS

    @classmethod
    def generate(cls, config: SubsetConfig) -> list[Callable]:
        """Generate MMPretrain transforms from the configuration."""
        transforms = super().generate(config)

        cls._check_mandatory_transforms(
            transforms,
            mandatory_transforms={PackInputs},
        )

        return transforms
