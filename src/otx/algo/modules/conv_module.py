# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation copied ConvModule of mmcv.cnn.bricks.ConvModule."""

# TODO(someone): Revisit mypy errors after deprecation of mmlab

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm as BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm as InstanceNorm

from otx.algo.utils.weight_init import constant_init, kaiming_init

from .activation import build_activation_layer
from .norm import build_norm_layer
from .padding import build_padding_layer

if TYPE_CHECKING:
    from torch.nn.modules.conv import _ConvNd as ConvNd


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon two build methods: `build_norm_layer()` and `build_activation_layer()`.

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
    """

    _abbr_ = "conv_block"
    _conv_nd: ConvNd

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
        norm_cfg: dict | None = None,
        act_cfg: dict | None = {"type": "ReLU"},  # noqa: B006
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        assert norm_cfg is None or isinstance(norm_cfg, dict)  # noqa: S101
        official_padding_mode = ["zeros", "circular"]
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode

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
        self.conv = self._conv_nd(
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
            norm_channels = out_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)  # type: ignore[arg-type]
            self.add_module(self.norm_name, norm)
            if self.with_bias and isinstance(norm, (BatchNorm, InstanceNorm)):
                warnings.warn("Unnecessary conv bias before batch/instance norm", stacklevel=1)
        else:
            self.norm_name = None  # type: ignore[assignment]

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore[union-attr]
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
            if self.with_activation and self.act_cfg["type"] == "LeakyReLU":  # type: ignore[index]
                nonlinearity = "leaky_relu"
                a = self.act_cfg.get("negative_slope", 0.01)  # type: ignore[union-attr]
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm_layer, 1, bias=0)

    def forward(self, x: Tensor, activate: bool = True, norm: bool = True) -> Tensor:
        """Forward pass of the ConvModule.

        Args:
            x (Tensor): Input tensor.
            activate (bool, optional): Whether to apply activation. Defaults to True.
            norm (bool, optional): Whether to apply normalization. Defaults to True.

        Returns:
            Tensor: Output tensor.
        """
        if self.with_explicit_padding:
            x = self.padding_layer(x)
        x = self.conv(x)
        if norm and self.with_norm:
            x = self.norm_layer(x)  # type: ignore[misc]
        if activate and self.with_activation:
            x = self.activate(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.

    See https://arxiv.org/pdf/1704.04861.pdf for details.

    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers. It should be
    noted that there will be norm/activation layer in the depthwise conv block
    if `norm_cfg` and `act_cfg` are specified.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``. Default: 1.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``. Default: 0.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``. Default: 1.
        norm_cfg (dict): Default norm config for both depthwise ConvModule and
            pointwise ConvModule. Default: None.
        act_cfg (dict): Default activation config for both depthwise ConvModule
            and pointwise ConvModule. Default: dict(type='ReLU').
        dw_norm_cfg (dict): Norm config of depthwise ConvModule. If it is
            None, it will be the same as `norm_cfg`. Default: None.
        dw_act_cfg (dict): Activation config of depthwise ConvModule. If it is
            None, it will be the same as `act_cfg`. Default: None.
        pw_norm_cfg (dict): Norm config of pointwise ConvModule. If it is
            None, it will be the same as `norm_cfg`. Default: None.
        pw_act_cfg (dict): Activation config of pointwise ConvModule. If it is
            None, it will be the same as `act_cfg`. Default: None.
        kwargs (optional): Other shared arguments for depthwise and pointwise
            ConvModule. See ConvModule for ref.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        dw_norm_cfg: dict | None = None,
        dw_act_cfg: dict | None = None,
        pw_norm_cfg: dict | None = None,
        pw_act_cfg: dict | None = None,
        **kwargs,
    ):
        if act_cfg is None:
            act_cfg = {"type": "ReLU"}

        super().__init__()
        if "groups" in kwargs:
            msg = "groups should not be specified in DepthwiseSeparableConvModule."
            raise ValueError(msg)

        # if norm/activation config of depthwise/pointwise Conv2dModule is not
        # specified, use default config.
        dw_norm_cfg = dw_norm_cfg or norm_cfg
        dw_act_cfg = dw_act_cfg or act_cfg
        pw_norm_cfg = pw_norm_cfg or norm_cfg
        pw_act_cfg = pw_act_cfg or act_cfg

        # depthwise convolution
        self.depthwise_conv = Conv2dModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
            **kwargs,
        )

        self.pointwise_conv = Conv2dModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)


class Conv2dModule(ConvModule):
    """A conv2d block that bundles conv/norm/activation layers."""

    _conv_nd = nn.Conv2d


class Conv3dModule(ConvModule):
    """A conv3d block that bundles conv/norm/activation layers."""

    _conv_nd = nn.Conv3d
