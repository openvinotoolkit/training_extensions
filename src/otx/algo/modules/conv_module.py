# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation copied ConvModule of mmcv.cnn.bricks.ConvModule."""
# TODO(someone): Revisit mypy errors after deprecation of mmlab
# mypy: ignore-errors
from __future__ import annotations

import warnings
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm as BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm as InstanceNorm

from otx.algo.utils.weight_init import constant_init, kaiming_init

from .activation import build_activation_layer
from .conv import build_conv_layer
from .norm import build_norm_layer
from .padding import build_padding_layer

if TYPE_CHECKING:
    from torch.nn.modules.conv import _ConvNd as ConvNd


def efficient_conv_bn_eval_forward(bn: BatchNorm, conv: ConvNd, x: torch.Tensor) -> torch.Tensor:
    """Implementation based on https://arxiv.org/abs/2305.11624.

    "Tune-Mode ConvBN Blocks For Efficient Transfer Learning"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for training as well. It reduces memory and computation cost.

    Args:
        bn (_BatchNorm): a BatchNorm module.
        conv (nn._ConvNd): a conv module
        x (torch.Tensor): Input feature map.
    """
    # These lines of code are designed to deal with various cases
    # like bn without affine transform, and conv without bias
    weight_on_the_fly = conv.weight
    bias_on_the_fly = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_var)

    bn_weight = bn.weight if bn.weight is not None else torch.ones_like(bn.running_var)

    bn_bias = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_var)

    # shape of [C_out, 1, 1, 1] in Conv2d
    weight_coeff = torch.rsqrt(bn.running_var + bn.eps).reshape([-1] + [1] * (len(conv.weight.shape) - 1))
    # shape of [C_out, 1, 1, 1] in Conv2d
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # shape of [C_out, C_in, k, k] in Conv2d
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # shape of [C_out] in Conv2d
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * (bias_on_the_fly - bn.running_mean)

    return conv._conv_forward(x, weight_on_the_fly, bias_on_the_fly)  # noqa: SLF001


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
        efficient_conv_bn_eval (bool): Whether use efficient conv when the
            consecutive bn is in eval mode (either training or testing), as
            proposed in https://arxiv.org/abs/2305.11624 . Default: `False`.
    """

    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels: int | tuple[int, ...],
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool | str = "auto",
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = {"type": "ReLU"},  # noqa: B006
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = "zeros",
        order: tuple = ("conv", "norm", "act"),
        efficient_conv_bn_eval: bool = False,
    ):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)  # noqa: S101
        assert norm_cfg is None or isinstance(norm_cfg, dict)  # noqa: S101
        official_padding_mode = ["zeros", "circular"]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple)  # noqa: S101
        assert len(self.order) == 3  # noqa: S101
        assert set(order) == {"conv", "norm", "act"}  # noqa: S101

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = {"type": padding_mode}
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            norm_channels = out_channels if order.index("norm") > order.index("conv") else in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
            if self.with_bias and isinstance(norm, (BatchNorm, InstanceNorm)):
                warnings.warn("Unnecessary conv bias before batch/instance norm", stacklevel=1)
        else:
            self.norm_name = None

        self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_["type"] not in [
                "Tanh",
                "PReLU",
                "Sigmoid",
                "HSigmoid",
                "Swish",
                "GELU",
                "SiLU",
            ]:
                act_cfg_.setdefault("inplace", inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm_layer(self) -> nn.Module | None:
        """Get the normalization layer.

        Returns:
            nn.Module | None: The normalization layer.
        """
        if self.norm_name:
            return getattr(self, self.norm_name)
        return None

    def init_weights(self) -> None:
        """Init weights function for ConvModule."""
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and self.act_cfg["type"] == "LeakyReLU":
                nonlinearity = "leaky_relu"
                a = self.act_cfg.get("negative_slope", 0.01)
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm_layer, 1, bias=0)

    def forward(self, x: torch.Tensor, activate: bool = True, norm: bool = True) -> torch.Tensor:
        """Forward pass of the ConvModule.

        Args:
            x (torch.Tensor): Input tensor.
            activate (bool, optional): Whether to apply activation. Defaults to True.
            norm (bool, optional): Whether to apply normalization. Defaults to True.

        Returns:
            torch.Tensor: Output tensor.
        """
        layer_index = 0
        while layer_index < len(self.order):
            layer = self.order[layer_index]
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                # if the next operation is norm and we have a norm layer in
                # eval mode and we have enabled `efficient_conv_bn_eval` for
                # the conv operator, then activate the optimized forward and
                # skip the next norm operator since it has been fused
                if (
                    layer_index + 1 < len(self.order)
                    and self.order[layer_index + 1] == "norm"
                    and norm
                    and self.with_norm
                    and not self.norm_layer.training
                    and self.efficient_conv_bn_eval_forward is not None
                ):
                    self.conv.forward = partial(self.efficient_conv_bn_eval_forward, self.norm_layer, self.conv)
                    layer_index += 1
                    x = self.conv(x)
                    del self.conv.forward
                else:
                    x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm_layer(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
            layer_index += 1
        return x

    def turn_on_efficient_conv_bn_eval(self, efficient_conv_bn_eval: bool = True) -> None:
        """Turn on the efficient convolution batch normalization evaluation.

        Args:
            efficient_conv_bn_eval (bool, optional): Whether to enable efficient convolution
                batch normalization evaluation. Defaults to True.
        """
        # efficient_conv_bn_eval works for conv + bn
        # with `track_running_stats` option
        if (
            efficient_conv_bn_eval
            and self.norm_layer
            and isinstance(self.norm_layer, BatchNorm)
            and self.norm_layer.track_running_stats
        ):
            self.efficient_conv_bn_eval_forward = efficient_conv_bn_eval_forward
        else:
            self.efficient_conv_bn_eval_forward = None

    @staticmethod
    def create_from_conv_bn(
        conv: ConvNd,
        bn: BatchNorm,
        efficient_conv_bn_eval: bool = True,
    ) -> ConvModule:
        """Create a ConvModule from a conv and a bn module."""
        self = ConvModule.__new__(ConvModule)
        super(ConvModule, self).__init__()

        self.conv_cfg = None
        self.norm_cfg = None
        self.act_cfg = None
        self.inplace = False
        self.with_spectral_norm = False
        self.with_explicit_padding = False
        self.order = ("conv", "norm", "act")

        self.with_norm = True
        self.with_activation = False
        self.with_bias = conv.bias is not None

        # build convolution layer
        self.conv = conv
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        self.norm_name, norm = "bn", bn
        self.add_module(self.norm_name, norm)

        self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)

        return self
