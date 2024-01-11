# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""EfficientNetB0 model implementation."""

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.classification import (
    MMPretrainHlabelClsModel,
    MMPretrainMulticlassClsModel,
    MMPretrainMultilabelClsModel,
)


class EfficientNetB0ForHLabelCls(MMPretrainHlabelClsModel):
    """EfficientNetB0 Model for hierarchical label classification task."""

    def __init__(self, num_classes: int, num_multiclass_heads: int, num_multilabel_classes: int) -> None:
        config = read_mmconfig(model_name="efficientnet_b0_light", subdir_name="hlabel_classification")
        config.head.num_multiclass_heads = num_multiclass_heads
        config.head.num_multilabel_classes = num_multilabel_classes
        super().__init__(num_classes=num_classes, config=config)


class EfficientNetB0ForMulticlassCls(MMPretrainMulticlassClsModel):
    """EfficientNetB0 Model for multi-label classification task."""

    def __init__(self, num_classes: int, light: bool = False) -> None:
        model_name = "efficientnet_b0_light" if light else "efficientnet_b0"
        config = read_mmconfig(model_name=model_name, subdir_name="multiclass_classification")
        super().__init__(num_classes=num_classes, config=config)


class EfficientNetB0ForMultilabelCls(MMPretrainMultilabelClsModel):
    """EfficientNetB0 Model for multi-class classification task."""

    def __init__(self, num_classes: int) -> None:
        config = read_mmconfig(model_name="efficientnet_b0_light", subdir_name="multilabel_classification")
        super().__init__(num_classes=num_classes, config=config)
