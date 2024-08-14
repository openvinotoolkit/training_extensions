# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""EfficientNet Module."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Literal

import torch
from pytorchcv.models.model_store import download_model
from torch import nn
from torch.nn import functional, init

from otx.algo.modules.activation import Swish, build_activation_layer
from otx.algo.modules.conv_module import Conv2dModule
from otx.algo.modules.norm import build_norm_layer
from otx.algo.utils.mmengine_utils import load_checkpoint_to_model

PRETRAINED_ROOT = "https://github.com/osmr/imgclsmob/releases/download/v0.0.364/"
pretrained_urls = {
    "efficientnet_b0": PRETRAINED_ROOT + "efficientnet_b0-0752-0e386130.pth.zip",
}


def conv1x1_block(
    in_channels: int,
    out_channels: int,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    groups: int = 1,
    bias: bool = False,
    use_bn: bool = True,
    bn_eps: float = 1e-5,
    activation: Callable[..., nn.Module] | None = nn.ReLU,
) -> Conv2dModule:
    """Conv block."""
    return Conv2dModule(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        normalization=build_norm_layer(nn.BatchNorm2d, num_features=out_channels, eps=bn_eps) if use_bn else None,
        activation=build_activation_layer(activation),
    )


def conv3x3_block(
    in_channels: int,
    out_channels: int,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = False,
    use_bn: bool = True,
    bn_eps: float = 1e-5,
    activation: Callable[..., nn.Module] | None = nn.ReLU,
) -> Conv2dModule:
    """Conv block."""
    return Conv2dModule(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        normalization=build_norm_layer(nn.BatchNorm2d, num_features=out_channels, eps=bn_eps) if use_bn else None,
        activation=build_activation_layer(activation),
    )


def dwconv3x3_block(
    in_channels: int,
    out_channels: int,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 1,
    dilation: int = 1,
    bias: bool = False,
    use_bn: bool = True,
    bn_eps: float = 1e-5,
    activation: Callable[..., nn.Module] | None = nn.ReLU,
) -> Conv2dModule:
    """Conv block."""
    return Conv2dModule(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        bias=bias,
        normalization=build_norm_layer(nn.BatchNorm2d, num_features=out_channels, eps=bn_eps) if use_bn else None,
        activation=build_activation_layer(activation),
    )


def dwconv5x5_block(
    in_channels: int,
    out_channels: int,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 2,
    dilation: int = 1,
    bias: bool = False,
    use_bn: bool = True,
    bn_eps: float = 1e-5,
    activation: Callable[..., nn.Module] | None = nn.ReLU,
) -> Conv2dModule:
    """Conv block."""
    return Conv2dModule(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        bias=bias,
        normalization=build_norm_layer(nn.BatchNorm2d, num_features=out_channels, eps=bn_eps) if use_bn else None,
        activation=build_activation_layer(activation),
    )


def round_channels(channels: float, divisor: int = 8) -> int:
    """Round weighted channel number (make divisible operation).

    Args:
        channels : int or float. Original number of channels.
        divisor : int, default 8. Alignment value.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


def calc_tf_padding(x: torch.Tensor, kernel_size: int, stride: int | tuple[int, int] = 1, dilation: int = 1) -> tuple:
    """Calculate TF-same like padding size.

    Args:
        x : tensor. Input tensor.
        kernel_size : int. Convolution window size.
        stride : int, default 1. Strides of the convolution.
        dilation : int, default 1. Dilation value for convolution layer.
    """
    height, width = x.size()[2:]
    oh = math.ceil(height / stride)
    ow = math.ceil(width / stride)
    pad_h = max((oh - 1) * stride + (kernel_size - 1) * dilation + 1 - height, 0)
    pad_w = max((ow - 1) * stride + (kernel_size - 1) * dilation + 1 - width, 0)
    return pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,'.

    https://arxiv.org/abs/1709.01507.

    Args:
        channels (int): Number of channels.
        reduction (int): Squeeze reduction value. Default to 16.
        mid_channels (int | None): Number of middle channels. Defaults to None.
        round_mid (bool): Whether to round middle channel number (make divisible by 8). Defaults to False.
        use_conv (bool): Whether to convolutional layers instead of fully-connected ones. Defaults to True.
        mid_activation (Callable[..., nn.Module]): Activation layer module after the first convolution.
            Defaults to ``nn.ReLU``.
        out_activation (Callable[..., nn.Module]): Activation layer module after the last convolution.
            Defaults to ``nn.Sigmoid``.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        mid_channels: int | None = None,
        round_mid: bool = False,
        use_conv: bool = True,
        mid_activation: Callable[..., nn.Module] = nn.ReLU,
        out_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ):
        super().__init__()
        self.use_conv = use_conv
        if mid_channels is None:
            mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=True,
            )
        else:
            self.fc1 = nn.Linear(in_features=channels, out_features=mid_channels)
        self.activ = mid_activation()
        if use_conv:
            self.conv2 = nn.Conv2d(
                in_channels=mid_channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=True,
            )
        else:
            self.fc2 = nn.Linear(in_features=mid_channels, out_features=channels)
        self.sigmoid = out_activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        w = self.pool(x)
        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        return x * w


class EffiDwsConvUnit(nn.Module):
    """EfficientNet specific depthwise separable conv block/unit with BatchNorms and activations at each conv.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int, int]): Strides of the second convolution layer.
        bn_eps (float): Small float added to variance in Batch norm.
        activation (Callable[..., nn.Module]): Activation layer module.
        tf_mode (bool): Whether to use TF-like mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int],
        bn_eps: float,
        activation: Callable[..., nn.Module],
        tf_mode: bool,
    ):
        super().__init__()
        self.tf_mode = tf_mode
        self.residual = (in_channels == out_channels) and (stride == 1)

        self.dw_conv = dwconv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            padding=(0 if tf_mode else 1),
            bn_eps=bn_eps,
            activation=activation,
        )
        self.se = SEBlock(channels=in_channels, reduction=4, mid_activation=activation)
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        if self.residual:
            identity = x
        if self.tf_mode:
            x = functional.pad(x, pad=calc_tf_padding(x, kernel_size=3))
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.pw_conv(x)
        if self.residual:
            x = x + identity
        return x


class EffiInvResUnit(nn.Module):
    """EfficientNet inverted residual unit.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int | tuple[int, int]): Convolution window size.
        stride (int | tuple[int, int]): Strides of the second convolution layer.
        exp_factor (int): Factor for expansion of channels.
        se_factor (int): SE reduction factor for each unit.
        bn_eps (float): Small float added to variance in Batch norm.
        activation (Callable[..., nn.Module]): Activation layer module.
        tf_mode (bool): Whether to use TF-like mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | tuple[int, int],
        exp_factor: int,
        se_factor: int,
        bn_eps: float,
        activation: Callable[..., nn.Module],
        tf_mode: bool,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.tf_mode = tf_mode
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_se = se_factor > 0
        mid_channels = in_channels * exp_factor
        dwconv_block_fn = dwconv3x3_block if kernel_size == 3 else dwconv5x5_block

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bn_eps=bn_eps,
            activation=activation,
        )
        self.conv2 = dwconv_block_fn(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            padding=(0 if tf_mode else kernel_size // 2),
            bn_eps=bn_eps,
            activation=activation,
        )
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=(exp_factor * se_factor),
                mid_activation=activation,
            )
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        if self.residual:
            identity = x
        x = self.conv1(x)
        if self.tf_mode:
            x = functional.pad(
                x,
                pad=calc_tf_padding(x, kernel_size=self.kernel_size, stride=self.stride),
            )
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class EffiInitBlock(nn.Module):
    """EfficientNet specific initial block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bn_eps (float): Small float added to variance in Batch norm.
        activation (Callable[..., nn.Module] | None): Activation layer module.
        tf_mode (bool): Whether to use TF-like mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_eps: float,
        activation: Callable[..., nn.Module] | None,
        tf_mode: bool,
    ):
        super().__init__()
        self.tf_mode = tf_mode

        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            padding=(0 if tf_mode else 1),
            bn_eps=bn_eps,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        if self.tf_mode:
            x = functional.pad(x, pad=calc_tf_padding(x, kernel_size=3, stride=2))
        return self.conv(x)


class EfficientNet(nn.Module):
    """EfficientNet.

    Args:
        channels : list of list of int. Number of output channels for each unit.
        init_block_channels : int. Number of output channels for initial unit.
        final_block_channels : int. Number of output channels for the final block of the feature extractor.
        kernel_sizes : list of list of int. Number of kernel sizes for each unit.
        strides_per_stage : list int. Stride value for the first unit of each stage.
        expansion_factors : list of list of int. Number of expansion factors for each unit.
        tf_mode : bool, default False. Whether to use TF-like mode.
        bn_eps : float, default 1e-5. Small float added to variance in Batch norm.
        in_channels : int, default 3. Number of input channels.
        in_size : tuple of two ints, default (224, 224). Spatial size of the expected input image.
        dropout_cls : dict, dropout configurations.
        pooling_type : str, default 'avg'. Pooling type to use.
        bn_eval : bool, default False. Whether to use BatchNorm eval mode.
        bn_frozen : bool, default False. Whether to freeze BatchNorm parameters.
        instance_norm_first : bool, default False. Whether to use instance normalization first.
        pretrained : bool, default False. Whether to load ImageNet pre-trained weights.
    """

    def __init__(
        self,
        channels: list[list[int]],
        init_block_channels: int,
        final_block_channels: int,
        kernel_sizes: list[list[int]],
        strides_per_stage: list[int],
        expansion_factors: list[list[int]],
        tf_mode: bool = False,
        bn_eps: float = 1e-5,
        in_channels: int = 3,
        in_size: tuple[int, int] = (224, 224),
        dropout_cls: dict | None = None,
        pooling_type: str | None = "avg",
        bn_eval: bool = False,
        bn_frozen: bool = False,
        instance_norm_first: bool = False,
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = 1000
        self.pretrained = pretrained
        self.in_size = in_size
        self.input_IN = nn.InstanceNorm2d(3, affine=True) if instance_norm_first else None
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.pooling_type = pooling_type
        self.num_features = self.num_head_features = final_block_channels
        activation = Swish
        self.features = nn.Sequential()
        self.features.add_module(
            "init_block",
            EffiInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_eps=bn_eps,
                activation=activation,
                tf_mode=tf_mode,
            ),
        )
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            kernel_sizes_per_stage = kernel_sizes[i]
            expansion_factors_per_stage = expansion_factors[i]
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                kernel_size = kernel_sizes_per_stage[j]
                expansion_factor = expansion_factors_per_stage[j]
                stride = strides_per_stage[i] if (j == 0) else 1
                if i == 0:
                    stage.add_module(
                        f"unit{j + 1}",
                        EffiDwsConvUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            stride=stride,
                            bn_eps=bn_eps,
                            activation=activation,
                            tf_mode=tf_mode,
                        ),
                    )
                else:
                    stage.add_module(
                        f"unit{j + 1}",
                        EffiInvResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            exp_factor=expansion_factor,
                            se_factor=4,
                            bn_eps=bn_eps,
                            activation=activation,
                            tf_mode=tf_mode,
                        ),
                    )
                in_channels = out_channels
            self.features.add_module(f"stage{i+1}", stage)
            # activation = activation if self.loss == 'softmax': else lambda: nn.PReLU(init=0.25)
        self.features.add_module(
            "final_block",
            conv1x1_block(
                in_channels=in_channels,
                out_channels=final_block_channels,
                bn_eps=bn_eps,
                activation=activation,
            ),
        )
        self._init_params()

    def _init_params(self) -> None:
        for module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_featuremaps: bool = False,
        get_embeddings: bool = False,
    ) -> tuple | list[torch.Tensor] | torch.Tensor:
        """Forward."""
        if self.input_IN is not None:
            x = self.input_IN(x)

        y = self.features(x)
        if return_featuremaps:
            return (y,)

        glob_features = self._glob_feature_vector(y, self.pooling_type, reduce_dims=False)

        logits = self.output(glob_features.view(x.shape[0], -1))

        if not self.training and self.classification:
            return [logits]

        if get_embeddings:
            out_data = [logits, glob_features.view(x.shape[0], -1)]
        elif self.loss in ["softmax", "am_softmax"]:
            out_data = logits if self.lr_finder.enable and self.lr_finder.mode == "automatic" else [logits]

        elif self.loss in ["triplet"]:
            out_data = [logits, glob_features]
        else:
            msg = f"Unsupported loss: {self.loss}"
            raise KeyError(msg)

        if self.lr_finder.enable and self.lr_finder.mode == "automatic":
            return out_data
        return tuple(out_data)


EFFICIENTNET_VERSION = Literal["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]


class OTXEfficientNet(EfficientNet):
    """Create EfficientNet model with specific parameters.

    Args:
        version : str. Version of EfficientNet ('b0'...'b8').
        in_size : tuple of two ints. Spatial size of the expected input image.
    """

    def __init__(self, version: EFFICIENTNET_VERSION, input_size: tuple[int, int] | None = None, **kwargs):
        self.model_name = "efficientnet_" + version

        if version == "b0":
            in_size = (224, 224)
            depth_factor = 1.0
            width_factor = 1.0
        elif version == "b1":
            in_size = (240, 240)
            depth_factor = 1.1
            width_factor = 1.0
        elif version == "b2":
            in_size = (260, 260)
            depth_factor = 1.2
            width_factor = 1.1
        elif version == "b3":
            in_size = (300, 300)
            depth_factor = 1.4
            width_factor = 1.2
        elif version == "b4":
            in_size = (380, 380)
            depth_factor = 1.8
            width_factor = 1.4
        elif version == "b5":
            in_size = (456, 456)
            depth_factor = 2.2
            width_factor = 1.6
        elif version == "b6":
            in_size = (528, 528)
            depth_factor = 2.6
            width_factor = 1.8
        elif version == "b7":
            in_size = (600, 600)
            depth_factor = 3.1
            width_factor = 2.0
        elif version == "b8":
            in_size = (672, 672)
            depth_factor = 3.6
            width_factor = 2.2
        else:
            msg = f"Unsupported EfficientNet version {version}"
            raise ValueError(msg)

        if input_size is not None:
            in_size = input_size

        init_block_channels = 32
        layers = [1, 2, 2, 3, 3, 4, 1]
        downsample = [1, 1, 1, 1, 0, 1, 0]
        channels_per_layers = [16, 24, 40, 80, 112, 192, 320]
        expansion_factors_per_layers = [1, 6, 6, 6, 6, 6, 6]
        kernel_sizes_per_layers = [3, 3, 5, 3, 5, 5, 3]
        _strides_per_stage = [1, 2, 2, 2, 1, 2, 1]
        final_block_channels = 1280

        layers = [int(math.ceil(li * depth_factor)) for li in layers]
        channels_per_layers = [round_channels(ci * width_factor) for ci in channels_per_layers]

        from functools import reduce

        channels: list = reduce(
            lambda x, y: [*x, [y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
            zip(channels_per_layers, layers, downsample),
            [],
        )
        kernel_sizes: list = reduce(
            lambda x, y: [*x, [y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
            zip(kernel_sizes_per_layers, layers, downsample),
            [],
        )
        expansion_factors: list = reduce(
            lambda x, y: [*x, [y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
            zip(expansion_factors_per_layers, layers, downsample),
            [],
        )
        strides_per_stage: list = reduce(
            lambda x, y: [*x, [y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
            zip(_strides_per_stage, layers, downsample),
            [],
        )
        strides_per_stage = [si[0] for si in strides_per_stage]

        init_block_channels = round_channels(init_block_channels * width_factor)

        if width_factor > 1.0:
            final_block_channels = round_channels(final_block_channels * width_factor)

        super().__init__(
            channels=channels,
            init_block_channels=init_block_channels,
            final_block_channels=final_block_channels,
            kernel_sizes=kernel_sizes,
            strides_per_stage=strides_per_stage,
            expansion_factors=expansion_factors,
            dropout_cls={"dist": "none"},
            tf_mode=False,
            bn_eps=1e-5,
            in_size=in_size,
            **kwargs,
        )
        self.init_weights(self.pretrained)

    def forward(self, x: torch.Tensor, return_featuremaps: bool = True, get_embeddings: bool = False) -> torch.Tensor:
        """Forward."""
        return super().forward(x, return_featuremaps=return_featuremaps, get_embeddings=get_embeddings)

    def init_weights(self, pretrained: bool | str | None = None) -> None:
        """Initialize weights."""
        if isinstance(pretrained, str) and Path(pretrained).exists():
            checkpoint = torch.load(pretrained, None)
            load_checkpoint_to_model(self, checkpoint)
            print(f"init weight - {pretrained}")
        elif pretrained:
            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            download_model(net=self, model_name=self.model_name, local_model_store_dir_path=str(cache_dir))
            print(f"Download model weight in {cache_dir!s}")
