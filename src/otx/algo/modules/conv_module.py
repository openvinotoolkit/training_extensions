from future import __annotations__

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = 'zeros',
        conv_cfg: Dict[str, str] = {'type': 'Conv2d'},
        norm_cfg: Dict[str, str | Any] | None = None,
        act_cfg: Dict[str, str | Any] | None = None,
    ) -> None:
        """
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel (int, optional): Size of the convolving kernel
            stride (int, optional): Stride of the convolution
            padding (int, optional): Zero-padding added to both sides of the input
            dilation (int, optional): Spacing between kernel elements
            groups (int, optional): Number of blocked connections from input channels to output channels
            bias (bool, optional): If True, adds a learnable bias to the output
            padding_mode (str, optional): 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
            conv_type (str, optional): '2d' or '3d'. Default: '2d'
            norm_cfg (Dict[str, str | Any] | None): Normalization configuration. Defaults to None.
            act_cfg (Dict[str, str | Any] | None): Activation configuration. Defaults to None.
        """
        super().__init__()
        if conv_cfg and conv_cfg["type"] == 'Conv2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation,
                                  groups=groups, bias=bias, padding_mode=padding_mode)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)

        self.bn, self.gn = None, None
        if norm_cfg:
            norm_type = norm_cfg["type"]
            if norm_type == 'GN':
                num_groups = norm_cfg['num_groups']
                self.gn = nn.GroupNorm(num_groups, out_channels)
            elif norm_type == 'BN':
                self.bn = nn.BatchNorm2d(out_channels) if conv_cfg["type"] == 'Conv2d' else nn.BatchNorm3d(out_channels)

        if act_cfg:
            act_type = act_cfg["type"]
            if act_type == 'ReLU':
                self.act = nn.ReLU(inplace=True)
            elif act_type == 'PReLU':
                self.act = nn.PReLU(**act_cfg["params"])
            elif act_type == 'Sigmoid':
                self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):

        x = self.conv(x)
        if self.gn:
            x = self.gn(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)

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
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        dw_act_cfg (dict): Activation config of depthwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        pw_norm_cfg (dict): Norm config of pointwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        pw_act_cfg (dict): Activation config of pointwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        kwargs (optional): Other shared arguments for depthwise and pointwise
            ConvModule. See ConvModule for ref.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | Tuple[int, int],
                 stride: int | Tuple[int, int] = 1,
                 padding: int | Tuple[int, int] = 0,
                 dilation: int | Tuple[int, int] = 1,
                 norm_cfg: Dict | None = None,
                 act_cfg: Dict = dict(type='ReLU'),
                 dw_norm_cfg: Dict | str = 'default',
                 dw_act_cfg: Dict | str = 'default',
                 pw_norm_cfg: Dict | str = 'default',
                 pw_act_cfg: Dict | str = 'default',
                 **kwargs):
        super().__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'

        # if norm/activation config of depthwise/pointwise ConvModule is not
        # specified, use default config.
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg  # type: ignore # noqa E501
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg  # type: ignore # noqa E501
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg

        # depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,  # type: ignore
            act_cfg=dw_act_cfg,  # type: ignore
            **kwargs)

        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,  # type: ignore
            act_cfg=pw_act_cfg,  # type: ignore
            **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
