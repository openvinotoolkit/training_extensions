# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation replaces the functionality of mmcv.cnn.bricks.norm.build_norm_layer."""
# TODO(someone): Revisit mypy errors after deprecation of mmlab
# mypy: ignore-errors

from __future__ import annotations

import inspect

from torch import nn
from torch.nn import SyncBatchNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from otx.algo.utils.mmengine_utils import is_tuple_of

# class GRN(nn.Module):
#     """Global Response Normalization Module.

#     Copy from mmpretrain.models.utils.norm

#     Come from `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
#     Autoencoders <http://arxiv.org/abs/2301.00808>`_

#     Args:
#         in_channels (int): The number of channels of the input tensor.
#         eps (float): a value added to the denominator for numerical stability.
#             Defaults to 1e-6.
#     """

#     def __init__(self, in_channels: int, eps: float = 1e-6):
#         super().__init__()
#         self.in_channels = in_channels
#         self.gamma = nn.Parameter(torch.zeros(in_channels))
#         self.beta = nn.Parameter(torch.zeros(in_channels))
#         self.eps = eps

#     def forward(self, x: torch.Tensor, data_format: str = 'channel_first') -> torch.Tensor:
#         """Forward method.

#         Args:
#             x (torch.Tensor): The input tensor.
#             data_format (str): The format of the input tensor. If
#                 ``"channel_first"``, the shape of the input tensor should be
#                 (B, C, H, W). If ``"channel_last"``, the shape of the input
#                 tensor should be (B, H, W, C). Defaults to "channel_first".
#         """
#         if data_format == 'channel_last':
#             gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
#             nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
#             x = self.gamma * (x * nx) + self.beta + x
#         elif data_format == 'channel_first':
#             gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
#             nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
#             x = self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(
#                 1, -1, 1, 1) + x
#         return x


# class LayerNorm2d(nn.LayerNorm):
#     """LayerNorm on channels for 2d images.

#     Copy from mmpretrain.models.utils.norm

#     Args:
#         num_channels (int): The number of channels of the input tensor.
#         eps (float): a value added to the denominator for numerical stability.
#             Defaults to 1e-5.
#         elementwise_affine (bool): a boolean value that when set to ``True``,
#             this module has learnable per-element affine parameters initialized
#             to ones (for weights) and zeros (for biases). Defaults to True.
#     """

#     def __init__(self, num_channels: int, **kwargs) -> None:
#         super().__init__(num_channels, **kwargs)
#         self.num_channels = self.normalized_shape[0]

#     def forward(self, x: torch.Tensor, data_format: str = 'channel_first') -> torch.Tensor:
#         """Forward method.

#         Args:
#             x (torch.Tensor): The input tensor.
#             data_format (str): The format of the input tensor. If
#                 ``"channel_first"``, the shape of the input tensor should be
#                 (B, C, H, W). If ``"channel_last"``, the shape of the input
#                 tensor should be (B, H, W, C). Defaults to "channel_first".
#         """
#         assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
#             f'(N, C, H, W), but got tensor with shape {x.shape}'
#         if data_format == 'channel_last':
#             x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
#                              self.eps)
#         elif data_format == 'channel_first':
#             x = x.permute(0, 2, 3, 1)
#             x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias,
#                              self.eps)
#             # If the output is discontiguous, it may cause some unexpected
#             # problem in the downstream tasks
#             x = x.permute(0, 3, 1, 2).contiguous()
#         return x


NORM_DICT = {
    "BN": nn.BatchNorm2d,
    "BN1d": nn.BatchNorm1d,
    "BN2d": nn.BatchNorm2d,
    "BN3d": nn.BatchNorm3d,
    "SyncBN": SyncBatchNorm,
    "GN": nn.GroupNorm,
    "LN": nn.LayerNorm,
    # "LN2d": LayerNorm2d,
    "IN": nn.InstanceNorm2d,
    "IN1d": nn.InstanceNorm1d,
    "IN2d": nn.InstanceNorm2d,
    "IN3d": nn.InstanceNorm3d,
    # "GRN": GRN,
}


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
    return "norm_layer"


def build_norm_layer(cfg: dict, num_features: int, postfix: int | str = "") -> tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        msg = "cfg must be a dict"
        raise TypeError(msg)
    if "type" not in cfg:
        msg = 'the cfg dict must contain the key "type"'
        raise KeyError(msg)
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    if inspect.isclass(layer_type):
        norm_layer = layer_type
    else:
        # Switch registry to the target scope. If `norm_layer` cannot be found
        # in the registry, fallback to search `norm_layer` in the
        # mmengine.MODELS.
        norm_layer = NORM_DICT.get(layer_type)
        if norm_layer is None:
            msg = f"Cannot find {norm_layer} in {NORM_DICT.keys()} "
            raise KeyError(msg)
    abbr = infer_abbr(norm_layer)

    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == "SyncBN" and hasattr(layer, "_specify_ddp_gpu_num"):
            layer._specify_ddp_gpu_num(1)  # noqa: SLF001
    else:
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def is_norm(layer: nn.Module, exclude: type | tuple | None = None) -> bool:
    """Check if a layer is a normalization layer.

    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type | tuple[type]): Types to be excluded.

    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude,)
        if not is_tuple_of(exclude, type):
            msg = f"'exclude' must be either None or type or a tuple of types, but got {type(exclude)}: {exclude}"
            raise TypeError(msg)

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)
