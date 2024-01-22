# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MobileNetV3 model implementation."""
from typing import Any

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.classification import (
    MMPretrainHlabelClsModel,
    MMPretrainMulticlassClsModel,
    MMPretrainMultilabelClsModel,
)

class MobileNetV3LoadWeightMixin:
    def convert_previous_otx_ckpt_impl(
        self, 
        state_dict: dict[str, Any], 
        label_type: str,
        add_prefix: str = "", 
    ):
        """Convert the previous OTX ckpt according to OTX2.0."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if key.startswith("classifier."):
                if "4" in key:
                    key = "head." + key.replace("4", "3")
                    if label_type == "multilabel":
                        val = val.t()
                else:
                    key = "head." + key
            elif key.startswith("act"):
                key = "head." + key
            elif not key.startswith("backbone."):
                key = "backbone." + key
            state_dict[add_prefix + key] = val
        return state_dict 
class MobileNetV3ForHLabelCls(MobileNetV3LoadWeightMixin, MMPretrainHlabelClsModel):
    """MobileNetV3 Model for hierarchical label classification task."""

    def __init__(self, num_classes: int, num_multiclass_heads: int, num_multilabel_classes: int) -> None:
        config = read_mmconfig(model_name="mobilenet_v3_large_light", subdir_name="hlabel_classification")
        config.head.num_multiclass_heads = num_multiclass_heads
        config.head.num_multilabel_classes = num_multilabel_classes
        super().__init__(num_classes=num_classes, config=config)

    def convert_previous_otx_ckpt(self, state_dict, add_prefix: str=""):
        """Convert the previous OTX ckpt according to OTX2.0."""
        return self.convert_previous_otx_ckpt_impl(state_dict, "hlabel", add_prefix)

class MobileNetV3ForMulticlassCls(MobileNetV3LoadWeightMixin, MMPretrainMulticlassClsModel):
    """MobileNetV3 Model for multi-label classification task."""

    def __init__(self, num_classes: int, light: bool = False) -> None:
        model_name = "mobilenet_v3_large_light" if light else "mobilenet_v3_large"
        config = read_mmconfig(model_name=model_name, subdir_name="multiclass_classification")
        super().__init__(num_classes=num_classes, config=config)

    def convert_previous_otx_ckpt(self, state_dict, add_prefix: str=""):
        """Convert the previous OTX ckpt according to OTX2.0."""
        return self.convert_previous_otx_ckpt_impl(state_dict, "multiclass", add_prefix)

class MobileNetV3ForMultilabelCls(MobileNetV3LoadWeightMixin, MMPretrainMultilabelClsModel):
    """MobileNetV3 Model for multi-class classification task."""

    def __init__(self, num_classes: int) -> None:
        config = read_mmconfig("mobilenet_v3_large_light", subdir_name="multilabel_classification")
        super().__init__(num_classes=num_classes, config=config)

    def convert_previous_otx_ckpt(self, state_dict, add_prefix: str=""):
        """Convert the previous OTX ckpt according to OTX2.0."""
        return self.convert_previous_otx_ckpt_impl(state_dict, "multilabel", add_prefix)