# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""EfficientNetB0 model implementation."""
from typing import Any

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.classification import (
    MMPretrainHlabelClsModel,
    MMPretrainMulticlassClsModel,
    MMPretrainMultilabelClsModel,
)

class EfficientNetB0LoadWeightMixin:
    def convert_previous_otx_ckpt_impl(
        self, 
        state_dict: dict[str, Any],
        label_type: str,
        add_prefix: str = "", 
    ):
        """Convert the previous OTX ckpt according to OTX2.0."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if key.startswith("features.") and "activ" not in key:
                key = "backbone." + key
            elif key.startswith("output."):
                key = key.replace("output", "head")
                if not label_type == "hlabel":
                    key = key.replace("asl", "fc")
                val = val.t()
            state_dict[add_prefix + key] = val
        return state_dict 

class EfficientNetB0ForHLabelCls(EfficientNetB0LoadWeightMixin, MMPretrainHlabelClsModel):
    """EfficientNetB0 Model for hierarchical label classification task."""

    def __init__(self, num_classes: int, num_multiclass_heads: int, num_multilabel_classes: int) -> None:
        config = read_mmconfig(model_name="efficientnet_b0_light", subdir_name="hlabel_classification")
        config.head.num_multiclass_heads = num_multiclass_heads
        config.head.num_multilabel_classes = num_multilabel_classes
        super().__init__(num_classes=num_classes, config=config)

    def convert_previous_otx_ckpt(self, state_dict, add_prefix: str=""):
        """Convert the previous OTX ckpt according to OTX2.0."""
        return self.convert_previous_otx_ckpt_impl(state_dict, "hlabel", add_prefix)

class EfficientNetB0ForMulticlassCls(EfficientNetB0LoadWeightMixin, MMPretrainMulticlassClsModel):
    """EfficientNetB0 Model for multi-label classification task."""

    def __init__(self, num_classes: int, light: bool = False) -> None:
        model_name = "efficientnet_b0_light" if light else "efficientnet_b0"
        config = read_mmconfig(model_name=model_name, subdir_name="multiclass_classification")
        super().__init__(num_classes=num_classes, config=config)
    
    def convert_previous_otx_ckpt(self, state_dict, add_prefix: str=""):
        """Convert the previous OTX ckpt according to OTX2.0."""
        return self.convert_previous_otx_ckpt_impl(state_dict, "multiclass", add_prefix)


class EfficientNetB0ForMultilabelCls(EfficientNetB0LoadWeightMixin, MMPretrainMultilabelClsModel):
    """EfficientNetB0 Model for multi-class classification task."""

    def __init__(self, num_classes: int) -> None:
        config = read_mmconfig(model_name="efficientnet_b0_light", subdir_name="multilabel_classification")
        super().__init__(num_classes=num_classes, config=config)

    def convert_previous_otx_ckpt(self, state_dict, add_prefix: str=""):
        """Convert the previous OTX ckpt according to OTX2.0."""
        return self.convert_previous_otx_ckpt_impl(state_dict, "multilabel", add_prefix)