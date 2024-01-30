# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MobileNetV3 model implementation."""
from typing import Any
from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.classification import (
    MMPretrainHlabelClsModel,
    MMPretrainMulticlassClsModel,
    MMPretrainMultilabelClsModel,
)


class MobileNetV3ForHLabelCls(MMPretrainHlabelClsModel):
    """MobileNetV3 Model for hierarchical label classification task."""

    def __init__(self, num_classes: int, num_multiclass_heads: int, num_multilabel_classes: int) -> None:
        config = read_mmconfig(model_name="mobilenet_v3_large_light", subdir_name="hlabel_classification")
        config.head.num_multiclass_heads = num_multiclass_heads
        config.head.num_multilabel_classes = num_multilabel_classes
        super().__init__(num_classes=num_classes, config=config)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parent_parameters = super()._export_parameters
        parent_parameters.update({"via_onnx": True})
        return parent_parameters

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_mobilenet_v3_ckpt(state_dict, "hlabel", add_prefix)


class MobileNetV3ForMulticlassCls(MMPretrainMulticlassClsModel):
    """MobileNetV3 Model for multi-label classification task."""

    def __init__(self, num_classes: int, light: bool = False) -> None:
        model_name = "mobilenet_v3_large_light" if light else "mobilenet_v3_large"
        config = read_mmconfig(model_name=model_name, subdir_name="multiclass_classification")
        super().__init__(num_classes=num_classes, config=config)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parent_parameters = super()._export_parameters
        parent_parameters.update({"via_onnx": True})
        return parent_parameters

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_mobilenet_v3_ckpt(state_dict, "multiclass", add_prefix)


class MobileNetV3ForMultilabelCls(MMPretrainMultilabelClsModel):
    """MobileNetV3 Model for multi-class classification task."""

    def __init__(self, num_classes: int) -> None:
        config = read_mmconfig("mobilenet_v3_large_light", subdir_name="multilabel_classification")
        super().__init__(num_classes=num_classes, config=config)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parent_parameters = super()._export_parameters
        parent_parameters.update({"via_onnx": True})
        return parent_parameters

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_mobilenet_v3_ckpt(state_dict, "multilabel", add_prefix)
