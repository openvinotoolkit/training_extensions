# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MobileNetV3 model implementation."""

from otx.algo.utils.mmconfig import read_mmconfig
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
        self.image_size = config["data_preprocessor"].get("size", (224, 224))
        super().__init__(num_classes=num_classes, config=config)

    def _configure_export_parameters(self) -> None:
        super()._configure_export_parameters()
        self.export_params["via_onnx"] = True


class MobileNetV3ForMulticlassCls(MMPretrainMulticlassClsModel):
    """MobileNetV3 Model for multi-label classification task."""

    def __init__(self, num_classes: int, light: bool = False) -> None:
        model_name = "mobilenet_v3_large_light" if light else "mobilenet_v3_large"
        config = read_mmconfig(model_name=model_name, subdir_name="multiclass_classification")
        self.image_size = config["data_preprocessor"].get("size", (224, 224))
        super().__init__(num_classes=num_classes, config=config)

    def _configure_export_parameters(self) -> None:
        super()._configure_export_parameters()
        self.export_params["via_onnx"] = True


class MobileNetV3ForMultilabelCls(MMPretrainMultilabelClsModel):
    """MobileNetV3 Model for multi-class classification task."""

    def __init__(self, num_classes: int) -> None:
        config = read_mmconfig("mobilenet_v3_large_light", subdir_name="multilabel_classification")
        self.image_size = config["data_preprocessor"].get("size", (224, 224))
        super().__init__(num_classes=num_classes, config=config)

    def _configure_export_parameters(self) -> None:
        super()._configure_export_parameters()
        self.export_params["via_onnx"] = True
