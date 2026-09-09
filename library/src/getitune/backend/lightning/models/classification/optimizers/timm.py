# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The optimizer factory for timm models, compatible with jsonargparse and the `OptimizerCallable` protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

from timm.optim import create_optimizer_v2

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer, params_t


_TRANSFORMER_PREFIXES = ("vit", "deit", "beit", "swin", "cait", "xcit", "maxvit", "coat", "twins", "pvt")


def _is_transformer_family(model_name: str) -> bool:
    """Check whether a timm model name belongs to a transformer-like family."""
    family = model_name.split("_")[0].split(".")[0].lower()
    return family.startswith(_TRANSFORMER_PREFIXES)


class TimmOptimizer:
    """`OptimizerCallable`-compatible factory for timm models, resolvable from a recipe.

    Selects AdamW for transformer-like backbones (ViT, DeiT, Swin, etc.) and SGD
    with momentum for convolutional backbones. `model_name` is not known when
    jsonargparse resolves the recipe (it is a sibling `init_arg` on the owning
    model, itself often `DYNAMIC`), so it is optional here and must be bound via
    `bind_model_name` before this optimizer is called.

    Args:
        lr: Base learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        model_name: Name of the timm backbone. May be left unset at
            construction time and populated later by the owning model.

    Example:
        >>> optimizer = TimmOptimizer(lr=0.0001, weight_decay=0.05)
        >>> optimizer.bind_model_name("vit_base_patch16_224")
        >>> opt = optimizer(model.parameters())
    """

    def __init__(self, lr: float, weight_decay: float, model_name: str | None = None) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_name = model_name

    def bind_model_name(self, model_name: str) -> None:
        """Bind the backbone name once resolved, unless one was already set explicitly.

        Args:
            model_name: Name of the timm backbone to select the optimizer family for.
        """
        if self.model_name is None:
            self.model_name = model_name

    def __call__(self, params: params_t) -> Optimizer:
        """Build the underlying optimizer for the given parameters.

        Args:
            params: Model parameters (or param groups) to optimize.

        Returns:
            The instantiated `torch.optim.Optimizer`.

        Raises:
            ValueError: If `model_name` was never bound.
        """
        if self.model_name is None:
            msg = "TimmOptimizer.model_name must be set (via bind_model_name) before it can be called."
            raise ValueError(msg)
        if _is_transformer_family(self.model_name):
            return create_optimizer_v2(
                params,
                opt="adamw",
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        return create_optimizer_v2(
            params,
            opt="sgd",
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
