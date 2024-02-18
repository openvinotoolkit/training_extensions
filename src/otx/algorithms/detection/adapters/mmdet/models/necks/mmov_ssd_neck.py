"""MMOVSSDNeck class for OMZ models."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, Optional, Union

import openvino.runtime as ov
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from mmdet.models.builder import NECKS
from torch import nn

from otx.core.ov.models.mmov_model import MMOVModel

# pylint: disable=too-many-arguments, too-many-locals


# FIXME: get rid of defined SSDNeck as this is a workaround for forked/deprecated mmdet
class SSDNeck(BaseModule):
    """Extra layers of SSD backbone to generate multi-scale feature maps.

    Args:
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (Sequence[int]): Number of output channels per scale.
        level_strides (Sequence[int]): Stride of 3x3 conv per level.
        level_paddings (Sequence[int]): Padding size of 3x3 conv per level.
        l2_norm_scale (float|None): L2 normalization layer init scale.
            If None, not use L2 normalization on the first input feature.
        last_kernel_size (int): Kernel size of the last conv layer.
            Default: 3.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        level_strides,
        level_paddings,
        l2_norm_scale=20.0,
        last_kernel_size=3,
        use_depthwise=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        init_cfg=None,
    ):
        if init_cfg is None:
            init_cfg = [
                dict(type="Xavier", distribution="uniform", layer="Conv2d"),
                dict(type="Constant", val=1, layer="BatchNorm2d"),
            ]
        super().__init__(init_cfg)
        assert len(out_channels) > len(in_channels)
        assert len(out_channels) - len(in_channels) == len(level_strides)
        assert len(level_strides) == len(level_paddings)
        assert in_channels == out_channels[: len(in_channels)]

        act_cfg = dict(type="ReLU") if act_cfg is None else act_cfg

        if l2_norm_scale:
            self.l2_norm = L2Norm(in_channels[0], l2_norm_scale)
            self.init_cfg += [dict(type="Constant", val=self.l2_norm.scale, override=dict(name="l2_norm"))]

        self.extra_layers = nn.ModuleList()
        extra_layer_channels = out_channels[len(in_channels) :]
        second_conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        for i, (out_channel, stride, padding) in enumerate(zip(extra_layer_channels, level_strides, level_paddings)):
            kernel_size = last_kernel_size if i == len(extra_layer_channels) - 1 else 3
            per_lvl_convs = nn.Sequential(
                ConvModule(
                    out_channels[len(in_channels) - 1 + i],
                    out_channel // 2,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
                second_conv(
                    out_channel // 2,
                    out_channel,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )
            self.extra_layers.append(per_lvl_convs)

    def forward(self, inputs):
        """Forward function."""
        outs = list(inputs)
        if hasattr(self, "l2_norm"):
            outs[0] = self.l2_norm(outs[0])

        feat = outs[-1]
        for layer in self.extra_layers:
            feat = layer(feat)
            outs.append(feat)
        return tuple(outs)


class L2Norm(nn.Module):
    """L2 normalization class."""

    def __init__(self, n_dims, scale=20.0, eps=1e-10):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super().__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        """Forward function."""
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) * x_float / norm).type_as(x)


@NECKS.register_module()
class MMOVSSDNeck(SSDNeck):
    """MMOVSSDNeck class for OMZ models."""

    def __init__(
        self,
        model_path_or_model: Union[str, ov.Model],
        weight_path: Optional[str] = None,
        inputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        outputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        init_weight: bool = False,
        verify_shape: bool = True,
    ):
        # dummy
        in_channels = (512, 1024)
        out_channels = (512, 1024, 512, 256, 256, 256)
        level_strides = (2, 2, 1, 1)
        level_paddings = (1, 1, 0, 0)
        l2_norm_scale = None
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            level_strides=level_strides,
            level_paddings=level_paddings,
            l2_norm_scale=l2_norm_scale,
        )

        self._model_path_or_model = model_path_or_model
        self._weight_path = weight_path
        self._init_weight = init_weight

        self.extra_layers = torch.nn.ModuleList()

        # TODO: Need to fix what exactly the types of inputs and outputs are.
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise ValueError("The type of inputs & outputs is invalid.")
        for input_e, output_e in zip(inputs["extra_layers"], outputs["extra_layers"]):
            if input_e and output_e:
                self.extra_layers.append(
                    MMOVModel(
                        model_path_or_model,
                        weight_path,
                        inputs=input_e,
                        outputs=output_e,
                        remove_normalize=False,
                        merge_bn=False,
                        paired_bn=False,
                        init_weight=init_weight,
                        verify_shape=verify_shape,
                    )
                )
            else:
                self.extra_layers.append(torch.nn.Identity())

        if "l2_norm" in inputs and "l2_norm" in outputs:
            for input_l2, output_l2 in zip(inputs["l2_norm"], outputs["l2_norm"]):
                self.l2_norm = MMOVModel(
                    model_path_or_model,
                    weight_path,
                    inputs=input_l2,
                    outputs=output_l2,
                    remove_normalize=False,
                    merge_bn=False,
                    paired_bn=False,
                    init_weight=init_weight,
                    verify_shape=verify_shape,
                )

    def init_weights(self, pretrained=None):  # pylint: disable=unused-argument
        """Initial weights of MMOVSSDNeck."""
        # TODO
        return
