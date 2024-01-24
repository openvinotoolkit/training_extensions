# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SSD object detector for the OTX detection."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.detection import MMDetCompatibleModel
from otx.core.utils.build import build_mm_model, modify_num_classes

if TYPE_CHECKING:
    import torch
    from mmengine.registry import Registry
    from omegaconf import DictConfig
    from torch import device, nn


class SSD(MMDetCompatibleModel):
    """Detecion model class for SSD."""

    def __init__(self, num_classes: int, variant: Literal["mobilenetv2"]) -> None:
        model_name = f"ssd_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)

    def _create_model(self) -> nn.Module:
        from mmdet.models.data_preprocessors import (
            DetDataPreprocessor as _DetDataPreprocessor,
        )
        from mmdet.registry import MODELS
        from mmengine.registry import MODELS as MMENGINE_MODELS

        # NOTE: For the history of this monkey patching, please see
        # https://github.com/openvinotoolkit/training_extensions/issues/2743
        @MMENGINE_MODELS.register_module(force=True)
        class DetDataPreprocessor(_DetDataPreprocessor):
            @property
            def device(self) -> device:
                try:
                    buf = next(self.buffers())
                except StopIteration:
                    return super().device
                else:
                    return buf.device

        self.classification_layers = self.get_classification_layers(self.config, MODELS, "model.")
        return build_mm_model(self.config, MODELS, self.load_from)

    @staticmethod
    def get_classification_layers(
        config: DictConfig,
        model_registry: Registry,
        prefix: str,
    ) -> dict[str, dict[str, bool | int]]:
        """Return classification layer names by comparing two different number of classes models.

        Args:
            config (DictConfig): Config for building model.
            model_registry (Registry): Registry for building model.
            prefix (str): Prefix of model param name.
                Normally it is "model." since OTXModel set it's nn.Module model as self.model

        Return:
            dict[str, dict[str, int]]
            A dictionary contain classification layer's name and information.
            `use_bg` means whether SSD use background class. It if True if SSD use softmax loss, and
            it is False if SSD use cross entropy loss.
            `num_anchors` means number of anchors of layer. SSD have classification per each anchor,
            so we have to update every anchors.
        """
        sample_config = deepcopy(config)
        modify_num_classes(sample_config, 3)
        sample_model_dict = build_mm_model(sample_config, model_registry, None).state_dict()

        modify_num_classes(sample_config, 4)
        incremental_model_dict = build_mm_model(sample_config, model_registry, None).state_dict()

        classification_layers = {}
        for key in sample_model_dict:
            if sample_model_dict[key].shape != incremental_model_dict[key].shape:
                sample_model_dim = sample_model_dict[key].shape[0]
                if sample_model_dim % 3 != 0:
                    use_bg = True
                    num_anchors = int(sample_model_dim / 4)
                    classification_layers[prefix + key] = {"use_bg": use_bg, "num_anchors": num_anchors}
                else:
                    use_bg = False
                    num_anchors = int(sample_model_dim / 3)
                    classification_layers[prefix + key] = {"use_bg": use_bg, "num_anchors": num_anchors}
        return classification_layers

    def load_state_dict_pre_hook(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs) -> None:
        """Modify input state_dict according to class name matching before weight loading."""
        model2ckpt = self.map_class_names(self.model_classes, self.ckpt_classes)

        for param_name, info in self.classification_layers.items():
            model_param = self.state_dict()[param_name].clone()
            ckpt_param = state_dict[prefix + param_name]
            use_bg = info["use_bg"]
            num_anchors = info["num_anchors"]
            if use_bg:
                num_ckpt_classes = len(self.ckpt_classes) + 1
                num_model_classes = len(self.model_classes) + 1
            else:
                num_ckpt_classes = len(self.ckpt_classes)
                num_model_classes = len(self.model_classes)

            for anchor_idx in range(num_anchors):
                for model_dst, ckpt_dst in enumerate(model2ckpt):
                    if ckpt_dst >= 0:
                        # Copying only matched weight rows
                        model_param[anchor_idx * num_model_classes + model_dst].copy_(
                            ckpt_param[anchor_idx * num_ckpt_classes + ckpt_dst],
                        )
                if use_bg:
                    model_param[anchor_idx * num_model_classes + num_model_classes - 1].copy_(
                        ckpt_param[anchor_idx * num_ckpt_classes + num_ckpt_classes - 1],
                    )

            # Replace checkpoint weight by mixed weights
            state_dict[prefix + param_name] = model_param

    def load_from_otx_v1_ckpt(self, state_dict, add_prefix: str="model.model."):
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix) 