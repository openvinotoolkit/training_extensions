# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation modified ConvModule of mmcv.cnn.bricks.ConvModule."""

# TODO(someone): Revisit mypy errors after deprecation of mmlab

from __future__ import annotations

import inspect
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Callable

from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm as BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm as InstanceNorm

from otx.algo.utils.weight_init import constant_init, kaiming_init

from .norm import build_norm_layer
from .padding import build_padding_layer

if TYPE_CHECKING:
    from torch.nn.modules.conv import _ConvNd as ConvNd


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon a build method: `build_norm_layer()`.

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
        bias (bool | str): If specified as `auto`, it will be decided by the normalization.
            Bias will be set as True if `normalization` is None, otherwise False.
            Defaults to "auto".
        normalization (tuple[str, nn.Module] | None): Tuple of normalization layer name and module
            generated by ``build_norm_layer``.
            Defaults to None.
            TODO (sungchul): enable nn.Module standalone and return it with appropriate name
        activation (Callable[..., nn.Module] | nn.Module | None): Activation layer module.
            Defaults to ``nn.ReLU``.
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
        normalization: tuple[str, nn.Module] | None = None,
        activation: Callable[..., nn.Module] | nn.Module | None = nn.ReLU,
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        official_padding_mode = ["zeros", "circular"]
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode

        self.with_norm = normalization is not None
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

        self.norm_name: str | None = None
        if normalization is not None:
            self.norm_name, norm = normalization
            self.add_module(self.norm_name, norm)
            if self.with_bias and isinstance(norm, (BatchNorm, InstanceNorm)):
                warnings.warn("Unnecessary conv bias before batch/instance norm", stacklevel=1)

        # build activation layer
        if isinstance(activation, type):
            activation = activation()
        self.activation: nn.Module | None = activation
        self._with_activation: bool | None = None

        # Use msra init by default
        self.init_weights()

    @property
    def with_activation(self) -> bool:
        """Whether the conv module has activation."""
        if self._with_activation is not None:
            # src/otx/algo/segmentation/heads/fcn_head.py L144
            return self._with_activation
        return self.activation is not None

    @with_activation.setter
    def with_activation(self, value: bool) -> None:
        """Setter for with_activation.

        For src/otx/algo/segmentation/heads/fcn_head.py L144.
        """
        self._with_activation = value

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
            if self.with_activation and isinstance(self.activation, nn.LeakyReLU):
                nonlinearity = "leaky_relu"
                a = getattr(self.activation, "negative_slop", 0.01)
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
            x = self.activation(x)  # type: ignore[misc]
        return x


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.

    See https://arxiv.org/pdf/1704.04861.pdf for details.

    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers. It should be
    noted that there will be norm/activation layer in the depthwise conv block
    if `normalization` and `activation` are specified.

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
        normalization (tuple[str, nn.Module] | None): Normalization layer module
            for both depthwise ConvModule and pointwise ConvModule.
            Defaults to None.
        activation (nn.Module | None): Activation layer module
            for both depthwise ConvModule and pointwise ConvModule.
            Defaults to ``nn.ReLU``.
        dw_normalization (tuple[str, nn.Module] | None): Normalization layer module of depthwise ConvModule.
            If it is None, it will be the same as ``normalization``.
            If it is already instanstiated to nn.Module, it's parameters will be realigned with `in_channels`.
            Defaults to None.
        dw_activation (nn.Module | None): Activation layer module of depthwise ConvModule.
            If it is None, it will be the same as ``activation``.
            Defaults to None.
        pw_normalization (tuple[str, nn.Module] | None): Normalization layer module of pointwise ConvModule.
            If it is None, it will be the same as ``normalization``.
            If it is already instanstiated to nn.Module, it's parameters will be realigned with `in_channels`.
            Defaults to None.
        pw_activation (nn.Module | None): Activation layer module of pointwise ConvModule.
            If it is None, it will be the same as ``activation``.
            Defaults to None.
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
        normalization: tuple[str, nn.Module] | None = None,
        activation: nn.Module | None = nn.ReLU,
        dw_normalization: tuple[str, nn.Module] | None = None,
        dw_activation: nn.Module | None = None,
        pw_normalization: tuple[str, nn.Module] | None = None,
        pw_activation: nn.Module | None = None,
        **kwargs,
    ):
        super().__init__()
        if "groups" in kwargs:
            msg = "groups should not be specified in DepthwiseSeparableConvModule."
            raise ValueError(msg)

        def update_num_features(
            normalization: tuple[str, nn.Module] | None,
            num_channels: int,
        ) -> tuple[str, nn.Module] | None:
            if normalization is None:
                return normalization

            if normalization[1].num_features != num_channels:
                init_signature = inspect.signature(normalization[1].__init__)
                init_parameters = {
                    k: getattr(normalization[1], k) for k in init_signature.parameters if k in normalization[1].__dict__
                }
                num_channels_name = "num_channels" if isinstance(normalization[1], nn.GroupNorm) else "num_features"
                init_parameters.update({num_channels_name: num_channels})
                normalization[1].__init__(**init_parameters)
            return build_norm_layer(normalization, num_features=in_channels)

        # if norm/activation config of depthwise/pointwise Conv2dModule is not
        # specified, use default config.
        dw_normalization = dw_normalization or deepcopy(normalization)
        dw_normalization = update_num_features(dw_normalization, in_channels)
        pw_normalization = pw_normalization or deepcopy(normalization)
        pw_normalization = update_num_features(pw_normalization, out_channels)

        dw_activation = dw_activation or activation
        pw_activation = pw_activation or activation

        # depthwise convolution
        self.depthwise_conv = Conv2dModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            normalization=dw_normalization,
            activation=dw_activation,
            **kwargs,
        )

        self.pointwise_conv = Conv2dModule(
            in_channels,
            out_channels,
            1,
            normalization=pw_normalization,
            activation=pw_activation,
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
