# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation replaces the functionality of mmcv.cnn.bricks.norm.build_norm_layer."""
# TODO(someone): Revisit mypy errors after deprecation of mmlab
# mypy: ignore-errors

from __future__ import annotations

import inspect
from functools import partial
from typing import Callable, Literal

import torch
from torch import nn
from torch.nn import SyncBatchNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


class FrozenBatchNorm2d(nn.Module):
    """Copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py.

    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        """Initialize FrozenBatchNorm2d.

        Args:
            num_features (int): Number of input features.
            eps (float): Epsilon for batch norm. Defaults to 1e-5.
        """
        super().__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self) -> str:
        """Str representation."""
        return "{num_features}, eps={eps}".format(**self.__dict__)


AVAILABLE_NORM_LIST = [
    "BN",
    "BN1d",
    "BN2d",
    "BN3d",
    "SyncBN",
    "GN",
    "LN",
    "IN",
    "IN1d",
    "IN2d",
    "IN3d",
    "FBN",
]


def infer_abbr(class_type: type) -> str:
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        msg = f"class_type must be a type, but got {type(class_type)}"
        raise TypeError(msg)
    if hasattr(class_type, "_abbr_"):
        return class_type._abbr_  # noqa: SLF001
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return "in"
    if issubclass(class_type, _BatchNorm):
        return "bn"
    if issubclass(class_type, nn.GroupNorm):
        return "gn"
    if issubclass(class_type, nn.LayerNorm):
        return "ln"
    class_name = class_type.__name__.lower()
    if "batch" in class_name:
        return "bn"
    if "group" in class_name:
        return "gn"
    if "layer" in class_name:
        return "ln"
    if "instance" in class_name:
        return "in"
    return "norm"


def _get_norm_type_name(
    norm_callable: Callable[..., nn.Module],
    target: Literal["class", "name"] = "class",
) -> type | str:
    """Get class type or name of given normalization callable.

    Args:
        norm_callable (Callable[..., nn.Module]): Normalization layer module.
        target (Literal["class", "name"]): The target to retrieve class type or name from. Defaults to "class".

    Returns:
        (type | str): Class type or name of given normalization callable.

    """
    if target == "class":
        return norm_callable.func if isinstance(norm_callable, partial) else norm_callable
    return norm_callable.func.__name__ if isinstance(norm_callable, partial) else norm_callable.__name__


def build_norm_layer(
    norm_callable: Callable[..., nn.Module],
    num_features: int,
    postfix: int | str = "",
    layer_name: str | None = None,
    requires_grad: bool = True,
    eps: float = 1e-5,
    **kwargs,
) -> tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        norm_callable (Callable[..., nn.Module] | None): Normalization layer module.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.
        layer_name (str | None): The name of the layer. Defaults to None.
        requires_grad (bool): Whether stop gradient updates. Defaults to True.
        eps (float): A value added to the denominator for numerical stability. Defaults to 1e-5.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not callable(norm_callable):
        msg = f"norm_callable must be a callable, but got {type(norm_callable)}."
        raise TypeError(msg)

    if _get_norm_type_name(norm_callable, target="name") not in AVAILABLE_NORM_LIST:
        msg = f"Unsupported normalization: {norm_callable.func.__name__}."
        raise ValueError(msg)

    abbr = infer_abbr(_get_norm_type_name(norm_callable)) if layer_name is None else layer_name
    name = abbr + str(postfix)

    if (layer_type := _get_norm_type_name(norm_callable)) is not nn.GroupNorm:
        layer = norm_callable(num_features, eps=eps, **kwargs)
        if layer_type == SyncBatchNorm and hasattr(layer, "_specify_ddp_gpu_num"):
            layer._specify_ddp_gpu_num(1)  # noqa: SLF001
    else:
        layer = norm_callable(num_channels=num_features, eps=eps, **kwargs)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def is_norm(layer: nn.Module) -> bool:
    """Check if a layer is a normalization layer.

    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type | tuple[type]): Types to be excluded.

    Returns:
        bool: Whether the layer is a norm layer.
    """
    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)
