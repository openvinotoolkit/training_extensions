# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for loading pretrained weights into Lightning model wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, cast

if TYPE_CHECKING:
    from torch import nn

    from getitune.types import PathLike


class _SupportsPretrainedWeights(Protocol):
    """Protocol for classes that can load pretrained weights.

    Implementing classes expose a wrapped PyTorch module, identify the selected
    model variant, and provide checkpoint locations keyed by model name.

    Attributes:
        pretrained_urls: Mapping from model names to pretrained checkpoint paths or URLs.
        pretrained_weights_target: Target module that receives the pretrained weights.
        model: Wrapped PyTorch module.
        model_name: Name used to resolve the default pretrained checkpoint.
    """

    pretrained_urls: ClassVar[dict[str, str]]
    pretrained_weights_target: Literal["model", "backbone"]
    model: nn.Module
    model_name: str

    @property
    def pretrained_key_mapping(self) -> dict[str, str] | None:
        """Mapping used to rename checkpoint keys before loading pretrained weights."""
        ...


class PretrainedWeightsMixin:
    """Mixin for loading pretrained weights into Lightning model wrappers.

    Classes using this mixin must define ``model``, ``model_name``, and
    ``pretrained_urls``. By default, weights are loaded into ``model``. Set
    ``pretrained_weights_target`` to ``"backbone"`` to load them into
    ``model.backbone`` instead.
    """

    pretrained_weights_target: Literal["model", "backbone"] = "model"

    @property
    def pretrained_key_mapping(self) -> dict[str, str] | None:
        """Mapping used to rename checkpoint keys before loading pretrained weights."""
        return None

    def load_pretrained(self: _SupportsPretrainedWeights, weights: PathLike | None = None) -> None:
        """Load pretrained weights into the configured model target.

        Args:
            weights (PathLike | None): Path to the pretrained weights file. If None, uses default weights.
        """
        from getitune.backend.lightning.models.utils.utils import load_checkpoint

        if weights is None:
            weights = self.pretrained_urls[self.model_name]

        target = self.model
        if self.pretrained_weights_target == "backbone":
            backbone = getattr(self.model, "backbone", None)
            if backbone is None:
                msg = f"{type(self.model).__name__} does not expose a backbone."
                raise AttributeError(msg)
            target = cast("nn.Module", backbone)

        load_checkpoint(target, str(weights), key_mapping=self.pretrained_key_mapping)
