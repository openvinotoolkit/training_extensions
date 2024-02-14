# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DeitTiny model implementation."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.classification import (
    ExplainableOTXClsModel,
    MMPretrainHlabelClsModel,
    MMPretrainMulticlassClsModel,
    MMPretrainMultilabelClsModel,
)


class ExplainableDeit(ExplainableOTXClsModel):
    """Deit model which can attach a XAI hook."""

    def register_explain_hook(self) -> None:
        """Register explain hook."""
        from otx.algo.hooks.recording_forward_hook import ViTReciproCAMHook

        target_layernorm = self.get_target_layernorm()
        self.explain_hook = ViTReciproCAMHook.create_and_register_hook(
            target_layernorm,
            self.head_forward_fn,
            num_classes=self.num_classes,
        )

    def get_target_layernorm(self, final_norm: bool = True) -> torch.nn.Module:
        """Returns the first (out of two) layernorm layer from the last backbone layer."""
        layernorm_layers = [module for module in self.backbone.modules() if isinstance(module, torch.nn.LayerNorm)]
        target_layernorm_index = -2 - int(final_norm)
        return layernorm_layers[target_layernorm_index]

    @torch.no_grad()
    def head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward."""
        if not hasattr(self.backbone, "layers"):
            raise ValueError
        if not hasattr(self.backbone, "final_norm"):
            raise ValueError
        if not hasattr(self.model, "with_neck"):
            raise ValueError

        # Part of the last transformer_encoder block (except first LayerNorm)
        target_layer = self.backbone.layers[-1]
        x = x + target_layer.attn(x)
        x = target_layer.ffn(target_layer.norm2(x), identity=x)

        # Final LayerNorm and neck
        if self.backbone.final_norm:
            x = self.backbone.norm1(x)
        if self.model.with_neck:
            x = self.model.neck(x)

        # Head
        cls_token = x[:, 0]
        layer_output = [None, cls_token]
        logit = self.model.head.forward(layer_output)
        if isinstance(logit, list):
            logit = torch.from_numpy(np.array(logit))
        return logit

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DeitTinyForMultilabelCls."""
        return {"model_type": "transformer"}


class DeitTinyForHLabelCls(ExplainableDeit, MMPretrainHlabelClsModel):
    """DeitTiny Model for hierarchical label classification task."""

    def __init__(self, num_classes: int, num_multiclass_heads: int, num_multilabel_classes: int) -> None:
        self.num_multiclass_heads = num_multiclass_heads
        self.num_multilabel_classes = num_multilabel_classes

        config = read_mmconfig(model_name="deit_tiny", subdir_name="hlabel_classification")
        config.head.num_multiclass_heads = num_multiclass_heads
        config.head.num_multilabel_classes = num_multilabel_classes
        super().__init__(num_classes=num_classes, config=config)

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "multiclass", add_prefix)


class DeitTinyForMulticlassCls(ExplainableDeit, MMPretrainMulticlassClsModel):
    """DeitTiny Model for multi-label classification task."""

    def __init__(self, num_classes: int) -> None:
        config = read_mmconfig("deit_tiny", subdir_name="multiclass_classification")
        super().__init__(num_classes=num_classes, config=config)

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "multiclass", add_prefix)


class DeitTinyForMultilabelCls(ExplainableDeit, MMPretrainMultilabelClsModel):
    """DeitTiny Model for multi-class classification task."""

    def __init__(self, num_classes: int) -> None:
        config = read_mmconfig("deit_tiny", subdir_name="multilabel_classification")
        super().__init__(num_classes=num_classes, config=config)

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "multiclass", add_prefix)
