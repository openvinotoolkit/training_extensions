# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""EfficientNetV2 model implementation."""
from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.classification import (
    MMPretrainHlabelClsModel,
    MMPretrainMulticlassClsModel,
    MMPretrainMultilabelClsModel,
)


class EfficientNetV2ForHLabelCls(MMPretrainHlabelClsModel):
    """EfficientNetV2 Model for hierarchical label classification task."""

    def __init__(self, num_classes: int, num_multiclass_heads: int, num_multilabel_classes: int) -> None:
        self.num_multiclass_heads = num_multiclass_heads
        self.num_multilabel_classes = num_multilabel_classes

        config = read_mmconfig("efficientnet_v2_light", subdir_name="hlabel_classification")
        config.head.num_multiclass_heads = num_multiclass_heads
        config.head.num_multilabel_classes = num_multilabel_classes
        super().__init__(num_classes=num_classes, config=config)

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_v2_ckpt(state_dict, "hlabel", add_prefix)


class EfficientNetV2ForMulticlassCls(MMPretrainMulticlassClsModel):
    """EfficientNetV2 Model for multi-label classification task."""

    def __init__(self, num_classes: int, light: bool = False) -> None:
        model_name = "efficientnet_v2_light" if light else "efficientnet_v2"
        config = read_mmconfig(model_name=model_name, subdir_name="multiclass_classification")
        super().__init__(num_classes=num_classes, config=config)

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_v2_ckpt(state_dict, "multiclass", add_prefix)


class EfficientNetV2ForMultilabelCls(MMPretrainMultilabelClsModel):
    """EfficientNetV2 Model for multi-class classification task."""

    def __init__(self, num_classes: int) -> None:
        config = read_mmconfig("efficientnet_v2_light", subdir_name="multilabel_classification")
        super().__init__(num_classes=num_classes, config=config)

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_v2_ckpt(state_dict, "multilabel", add_prefix)
