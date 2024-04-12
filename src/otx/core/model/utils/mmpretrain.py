# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility files for the mmpretrain package."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Callable, Generic

import torch
from mmpretrain.models.utils import ClsDataPreprocessor as _ClsDataPreprocessor
from mmpretrain.registry import MODELS

from otx.algo.explain.explain_algo import ReciproCAM, get_feature_vector
from otx.core.data.entity.base import T_OTXBatchDataEntity, T_OTXBatchPredEntity
from otx.core.utils.build import build_mm_model, get_classification_layers

if TYPE_CHECKING:
    from mmpretrain.models.classifiers.image import ImageClassifier
    from mmpretrain.structures import DataSample
    from omegaconf import DictConfig
    from torch import Tensor, device, nn


@MODELS.register_module(force=True)
class ClsDataPreprocessor(_ClsDataPreprocessor):
    """Class for monkey patching preprocessor.

    NOTE: For the history of this monkey patching, please see
    https://github.com/openvinotoolkit/training_extensions/issues/2743
    """

    @property
    def device(self) -> device:
        """Which device being used."""
        try:
            buf = next(self.buffers())
        except StopIteration:
            return super().device
        else:
            return buf.device


def create_model(config: DictConfig, load_from: str | None = None) -> tuple[nn.Module, dict[str, dict[str, int]]]:
    """Create a model from mmpretrain Model registry.

    Args:
        config (DictConfig): Model configuration.
        load_from (str | None): Model weight file path.

    Returns:
        tuple[nn.Module, dict[str, dict[str, int]]]: Model instance and classification layers.
    """
    classification_layers = get_classification_layers(config, MODELS, "model.")
    return build_mm_model(config, MODELS, load_from), classification_layers


class ExplainableMixInMMPretrainModel(Generic[T_OTXBatchPredEntity, T_OTXBatchDataEntity]):
    """Mix-in class to support XAI feature for MM Pretrained models commonly.

    This is a mix-in class that cannot be used as standalone. The correct usage of it will be like
    ```python
    class MobileNetV3ForMulticlassCls(ExplainableMixInMMPretrainModel, MMPretrainMulticlassClsModel):
    ```
    """

    explain_mode: bool
    num_classes: int
    model: ImageClassifier

    @property
    def has_gap(self) -> bool:
        """Defines if GAP is used right after backbone.

        Note:
            Can be redefined at the model's level.
        """
        return True

    @torch.no_grad()
    def head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward.

        Note:
            Can be redefined at the model's level.
        """
        if (neck := getattr(self.model, "neck", None)) is None:
            raise ValueError
        if (head := getattr(self.model, "head", None)) is None:
            raise ValueError

        output = neck(x)
        return head([output])

    @staticmethod
    def _forward_explain_image_classifier(
        self: ImageClassifier,
        inputs: torch.Tensor,
        data_samples: list[DataSample] | None = None,
        mode: str = "tensor",
    ) -> dict[str, torch.Tensor]:
        """Forward func of the ImageClassifier instance, which located in ExplainableOTXClsModel().model.

        Note:
            Can be redefined at the model's level.
        """
        x = self.backbone(inputs)
        backbone_feat = x

        feature_vector = self.feature_vector_fn(backbone_feat)
        saliency_map = self.explain_fn(backbone_feat)

        if self.with_neck:
            x = self.neck(x)

        if mode == "tensor":
            logits = self.head(x) if self.with_head else x
        elif mode == "predict":
            logits = self.head.predict(x, data_samples)
        else:
            msg = f'Invalid mode "{mode}".'
            raise RuntimeError(msg)

        return {
            "logits": logits,
            "feature_vector": feature_vector,
            "saliency_map": saliency_map,
        }

    def get_explain_fn(self) -> Callable:
        """Returns explain function.

        Note:
            Can be redefined at the model's level.
        """
        explainer = ReciproCAM(
            self.head_forward_fn,
            num_classes=self.num_classes,
            optimize_gap=self.has_gap,
        )
        return explainer.func

    def _register(self) -> None:
        if getattr(self, "_registered", False):
            return
        self.model.feature_vector_fn = get_feature_vector
        self.model.explain_fn = self.get_explain_fn()
        self._registered = True

    def forward_explain(
        self,
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity:
        """Model forward function."""
        self._register()
        orig_model_forward = self.model.forward

        try:
            self.model.forward = types.MethodType(self._forward_explain_image_classifier, self.model)

            forward_func: Callable[[T_OTXBatchDataEntity], T_OTXBatchPredEntity] | None = getattr(self, "forward", None)

            if forward_func is None:
                msg = (
                    "This instance has no forward function. "
                    "Did you attach this mixin into a class derived from OTXModel?"
                )
                raise RuntimeError(msg)

            return forward_func(inputs)
        finally:
            self.model.forward = orig_model_forward

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            self._register()
            forward_explain = types.MethodType(self._forward_explain_image_classifier, self.model)
            return forward_explain(inputs=image, mode="tensor")

        return self.model.forward(inputs=image, mode="tensor")
