# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Implementation of MobileNetV3.

Original papers:
- 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from otx.algo.utils.mmengine_utils import load_checkpoint_to_model, load_from_http

pretrained_root = "https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/"
pretrained_urls = {
    "mobilenetv3_small": pretrained_root + "mobilenetv3-small-55df8e1f.pth?raw=true",
    "mobilenetv3_large": pretrained_root + "mobilenetv3-large-1cd25616.pth?raw=true",
    "mobilenetv3_large_075": pretrained_root + "mobilenetv3-large-0.75-9632d2a8.pth?raw=true",
}


class ModelInterface(nn.Module):
    """Model Interface."""

    def __init__(
        self,
        classification: bool = False,
        contrastive: bool = False,
        pretrained: bool = False,
        loss: str = "softmax",
        **kwargs,
    ):
        super().__init__()

        self.classification = classification
        self.contrastive = contrastive
        self.pretrained = pretrained
        self.classification_classes: dict = {}
        self.loss = loss
        self.is_ie_model = False
        if loss == "am_softmax":
            self.use_angle_simple_linear = True
        else:
            self.use_angle_simple_linear = False

    @staticmethod
    def _glob_feature_vector(x: torch.Tensor, mode: str, reduce_dims: bool = True) -> torch.Tensor:
        if mode == "avg":
            out = functional.adaptive_avg_pool2d(x, 1)
        elif mode == "max":
            out = functional.adaptive_max_pool2d(x, 1)
        elif mode == "avg+max":
            avg_pool = functional.adaptive_avg_pool2d(x, 1)
            max_pool = functional.adaptive_max_pool2d(x, 1)
            out = avg_pool + max_pool
        else:
            msg = f"Unknown pooling mode: {mode}"
            raise ValueError(msg)

        if reduce_dims:
            return out.view(x.size(0), -1)
        return out


class HSigmoid(nn.Module):
    """Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'.

    https://arxiv.org/abs/1905.02244.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return functional.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    """H-Swish activation function from 'Searching for MobileNetV3,'.

    https://arxiv.org/abs/1905.02244.

    Parameters:
        inplace : bool, Whether to use inplace version of the module.
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return x * functional.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SELayer(nn.Module):
    """SE layer."""

    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(make_divisible(channel // reduction, 8), channel),
            HSigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        # with no_nncf_se_layer_context():
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp: int, oup: int, stride: int, instance_norm_conv1: bool = False) -> nn.Sequential:
    """Conv 3x3 layer with batch-norm."""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup) if not instance_norm_conv1 else nn.InstanceNorm2d(oup, affine=True),
        HSwish(),
    )


def conv_1x1_bn(inp: int, oup: int, loss: str = "softmax") -> nn.Sequential:
    """Conv 1x1 layer with batch-norm."""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        HSwish() if loss == "softmax" else nn.PReLU(),
    )


def make_divisible(value: int | float, divisor: int, min_value: int | None = None, min_ratio: float = 0.9) -> int:
    """Make divisible function.

    This function rounds the channel number down to the nearest value that can
    be divisible by the divisor.

    Args:
        value (int | float): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int, optional): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel
            number to the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class InvertedResidual(nn.Module):
    """Inverted residual."""

    def __init__(self, inp: int, hidden_dim: int, oup: int, kernel_size: int, stride: int, use_se: bool, use_hs: bool):
        super().__init__()

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        if self.identity:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV3Base(ModelInterface):
    """Base model of MobileNetV3."""

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        in_channels: int = 3,
        input_size: tuple[int, int] = (224, 224),
        dropout_cls: nn.Module | None = None,
        pooling_type: str = "avg",
        feature_dim: int = 1280,
        instance_norm_first: bool = False,
        self_challenging_cfg: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_size = input_size
        self.num_classes = num_classes
        self.input_IN = nn.InstanceNorm2d(in_channels, affine=True) if instance_norm_first else None
        self.pooling_type = pooling_type
        self.self_challenging_cfg = self_challenging_cfg
        self.width_mult = width_mult
        self.dropout_cls = dropout_cls
        self.feature_dim = feature_dim

    def infer_head(self, x: torch.Tensor, skip_pool: bool = False) -> torch.Tensor:
        """Inference head."""
        raise NotImplementedError

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Extract features."""
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward."""
        if self.input_IN is not None:
            x = self.input_IN(x)

        return self.extract_features(x)


class MobileNetV3(MobileNetV3Base):
    """MobileNetV3."""

    def __init__(self, cfgs: list, mode: str, instance_norm_conv1: bool = False, **kwargs):
        super().__init__(**kwargs)
        # setting of inverted residual blocks
        self.cfgs = cfgs
        # building first layer
        input_channel = make_divisible(16 * self.width_mult, 8)
        stride = 1 if self.in_size[0] < 100 else 2
        layers = [conv_3x3_bn(3, input_channel, stride, instance_norm_conv1)]
        # building inverted residual blocks
        block = InvertedResidual
        flag = True
        output_channel: int | dict[str, int]
        for k, t, c, use_se, use_hs, s in self.cfgs:
            _s = s
            if (self.in_size[0] < 100) and (s == 2) and flag:
                _s = 1
                flag = False
            output_channel = make_divisible(c * self.width_mult, 8)
            exp_size = make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, _s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size, self.loss)
        output_channel = {"large": 1280, "small": 1024}
        output_channel = (
            make_divisible(output_channel[mode] * self.width_mult, 8) if self.width_mult > 1.0 else output_channel[mode]
        )
        self._initialize_weights()

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Extract features."""
        y = self.conv(self.features(x))
        return (y,)

    def infer_head(self, x: torch.Tensor, skip_pool: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference head."""
        glob_features = self._glob_feature_vector(x, self.pooling_type, reduce_dims=False) if not skip_pool else x

        logits = self.classifier(glob_features.view(x.shape[0], -1))
        return glob_features, logits

    def _initialize_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class OTXMobileNetV3(MobileNetV3):
    """MobileNetV3 model for OTX."""

    backbone_configs = {  # noqa: RUF012
        "small": [
            # k, t, c, SE, HS, s
            [3, 1, 16, 1, 0, 2],
            [3, 4.5, 24, 0, 0, 2],
            [3, 3.67, 24, 0, 0, 1],
            [5, 4, 40, 1, 1, 2],
            [5, 6, 40, 1, 1, 1],
            [5, 6, 40, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 6, 96, 1, 1, 2],
            [5, 6, 96, 1, 1, 1],
            [5, 6, 96, 1, 1, 1],
        ],
        "large": [
            # k, t, c, SE, HS, s
            [3, 1, 16, 0, 0, 1],
            [3, 4, 24, 0, 0, 2],
            [3, 3, 24, 0, 0, 1],
            [5, 3, 40, 1, 0, 2],
            [5, 3, 40, 1, 0, 1],
            [5, 3, 40, 1, 0, 1],
            [3, 6, 80, 0, 1, 2],
            [3, 2.5, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [5, 6, 160, 1, 1, 2],
            [5, 6, 160, 1, 1, 1],
            [5, 6, 160, 1, 1, 1],
        ],
    }

    def __init__(self, mode: str = "large", width_mult: float = 1.0, **kwargs):
        super().__init__(self.backbone_configs[mode], mode=mode, width_mult=width_mult, **kwargs)
        self.key = "mobilenetv3_" + mode
        if width_mult != 1.0:
            self.key = self.key + f"_{int(width_mult * 100):03d}"  # pylint: disable=consider-using-f-string
        self.init_weights(self.pretrained)

    def init_weights(self, pretrained: str | bool | None = None) -> None:
        """Initialize weights."""
        checkpoint = None
        if isinstance(pretrained, str) and Path(pretrained).exists():
            checkpoint = torch.load(pretrained, None)
            print(f"init weight - {pretrained}")
        elif pretrained is not None:
            checkpoint = load_from_http(pretrained_urls[self.key])
            print(f"init weight - {pretrained_urls[self.key]}")
        if checkpoint is not None:
            load_checkpoint_to_model(self, checkpoint)
