# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation of DepthwiseSeparableConvModule copied from mmcv.cnn.bricks.depthwise_separable_conv_module."""

from __future__ import annotations

from torch import Tensor, nn

from .conv_module import ConvModule


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
        assert "groups" not in kwargs, "groups should not be specified"  # noqa: S101

        # if norm/activation config of depthwise/pointwise ConvModule is not
        # specified, use default config.
        dw_norm_cfg = dw_norm_cfg or norm_cfg
        dw_act_cfg = dw_act_cfg or act_cfg
        pw_norm_cfg = pw_norm_cfg or norm_cfg
        pw_act_cfg = pw_act_cfg or act_cfg

        # depthwise convolution
        self.depthwise_conv = ConvModule(
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

        self.pointwise_conv = ConvModule(
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
