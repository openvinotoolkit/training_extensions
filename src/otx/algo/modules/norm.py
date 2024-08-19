# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation replaces the functionality of mmcv.cnn.bricks.norm.build_norm_layer."""
# TODO(someone): Revisit mypy errors after deprecation of mmlab
# mypy: ignore-errors

from __future__ import annotations

import inspect
from functools import partial
from typing import Any, Callable

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


AVAILABLE_NORMALIZATION_LIST = [
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    SyncBatchNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    FrozenBatchNorm2d,
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


def _get_norm_type(normalization: Callable[..., nn.Module]) -> type:
    """Get class type or name of given normalization callable.

    Args:
        normalization (Callable[..., nn.Module]): Normalization layer module.

    Returns:
        (type): Class type of given normalization callable.

    """
    return normalization.func if isinstance(normalization, partial) else normalization


def build_norm_layer(
    normalization: Callable[..., nn.Module] | tuple[str, nn.Module] | nn.Module,
    num_features: int,
    postfix: int | str = "",
    layer_name: str | None = None,
    requires_grad: bool = True,
    eps: float = 1e-5,
    **kwargs,
) -> tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        normalization (Callable[..., nn.Module] | tuple[str, nn.Module] | nn.Module | None):
            Normalization layer module. If tuple or None is given, return it as is.
            If nn.Module is given, return it with empty name string. If callable is given, create the layer.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.
        layer_name (str | None): The name of the layer.
            Defaults to None.
        requires_grad (bool): Whether stop gradient updates.
            Defaults to True.
        eps (float): A value added to the denominator for numerical stability.
            Defaults to 1e-5.

    Returns:
        (tuple[str, nn.Module] | None): The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if normalization is None or isinstance(normalization, tuple):
        # return tuple or None as is
        return normalization

    if isinstance(normalization, nn.Module):
        # return nn.Module with empty name string
        return "", normalization

    if not callable(normalization):
        msg = f"normalization must be a callable, but got {type(normalization)}."
        raise TypeError(msg)

    if isinstance(normalization, partial) and normalization.func.__name__ == "build_norm_layer":
        # add arguments to `normalization` and return it
        signature = inspect.signature(normalization.func)
        predefined_key_list: list[str] = ["kwargs"]

        # find keys according to normalization.args except for normalization type (index=0)
        predefined_key_list.extend(list(signature.parameters.keys())[: len(normalization.args)])

        # find keys according to normalization.keywords
        predefined_key_list.extend(list(normalization.keywords.keys()))

        # set the remaining parameters previously undefined in normalization
        _locals = locals()
        fn_kwargs: dict[str, Any] = {
            k: _locals.get(k, None) for k in signature.parameters if k not in predefined_key_list
        }

        # manually update kwargs
        fn_kwargs.update(_locals.get("kwargs", {}))
        return normalization(**fn_kwargs)

    if (layer_type := _get_norm_type(normalization)) not in AVAILABLE_NORMALIZATION_LIST:
        msg = f"Unsupported normalization: {layer_type.__name__}."
        raise ValueError(msg)

    # set norm name
    abbr = layer_name or infer_abbr(layer_type)
    name = abbr + str(postfix)

    # set norm layer
    if layer_type is not nn.GroupNorm:
        layer = normalization(num_features, eps=eps, **kwargs)
        if layer_type == SyncBatchNorm and hasattr(layer, "_specify_ddp_gpu_num"):
            layer._specify_ddp_gpu_num(1)  # noqa: SLF001
    else:
        layer = normalization(num_channels=num_features, eps=eps, **kwargs)

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
