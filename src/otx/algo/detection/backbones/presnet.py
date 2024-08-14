# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Presnet backbones, modified from https://github.com/lyuwenyu/RT-DETR."""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, ClassVar

import torch
from torch import nn

from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.conv_module import Conv2dModule
from otx.algo.modules.norm import build_norm_layer

__all__ = ["PResNet"]


class BasicBlock(nn.Module):
    """BasicBlock."""

    expansion = 1

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        stride: int,
        shortcut: bool,
        activation: Callable[..., nn.Module] | None = None,
        variant: str = "b",
        normalization: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            (
                                "conv",
                                Conv2dModule(
                                    ch_in,
                                    ch_out,
                                    1,
                                    1,
                                    normalization=build_norm_layer(normalization, num_features=ch_out),
                                    activation=None,
                                ),
                            ),
                        ],
                    ),
                )
            else:
                self.short = Conv2dModule(
                    ch_in,
                    ch_out,
                    1,
                    stride,
                    normalization=build_norm_layer(normalization, num_features=ch_out),
                    activation=None,
                )

        self.branch2a = Conv2dModule(
            ch_in,
            ch_out,
            3,
            stride,
            padding=1,
            normalization=build_norm_layer(normalization, num_features=ch_out),
            activation=activation,
        )
        self.branch2b = Conv2dModule(
            ch_out,
            ch_out,
            3,
            1,
            padding=1,
            normalization=build_norm_layer(normalization, num_features=ch_out),
            activation=None,
        )
        self.act = activation() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.branch2a(x)
        out = self.branch2b(out)
        short = x if self.shortcut else self.short(x)

        out = out + short

        return self.act(out)


class BottleNeck(nn.Module):
    """BottleNeck."""

    expansion = 4

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        stride: int,
        shortcut: bool,
        activation: Callable[..., nn.Module] | None = None,
        variant: str = "b",
        normalization: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        if variant == "a":
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out

        self.branch2a = Conv2dModule(
            ch_in,
            width,
            1,
            stride1,
            normalization=build_norm_layer(normalization, num_features=width),
            activation=build_activation_layer(activation),
        )
        self.branch2b = Conv2dModule(
            width,
            width,
            3,
            stride2,
            padding=1,
            normalization=build_norm_layer(normalization, num_features=width),
            activation=build_activation_layer(activation),
        )
        self.branch2c = Conv2dModule(
            width,
            ch_out * self.expansion,
            1,
            1,
            normalization=build_norm_layer(
                normalization,
                num_features=ch_out * self.expansion,
            ),
            activation=None,
        )

        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            (
                                "conv",
                                Conv2dModule(
                                    ch_in,
                                    ch_out * self.expansion,
                                    1,
                                    1,
                                    normalization=build_norm_layer(
                                        normalization,
                                        num_features=ch_out * self.expansion,
                                    ),
                                    activation=None,
                                ),
                            ),
                        ],
                    ),
                )
            else:
                self.short = Conv2dModule(
                    ch_in,
                    ch_out * self.expansion,
                    1,
                    stride,
                    normalization=build_norm_layer(
                        normalization,
                        num_features=ch_out * self.expansion,
                    ),
                    activation=None,
                )

        self.act = activation() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)
        short = x if self.shortcut else self.short(x)

        out = out + short

        return self.act(out)


class Blocks(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        ch_in: int,
        ch_out: int,
        count: int,
        stage_num: int,
        activation: Callable[..., nn.Module] | None = None,
        variant: str = "b",
        normalization: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in,
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=i != 0,
                    variant=variant,
                    activation=activation,
                    normalization=normalization,
                ),
            )

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class PResNet(BaseModule):
    """PResNet backbone.

    Args:
        depth (int): The depth of the PResNet backbone.
        variant (str): The variant of the PResNet backbone. Defaults to "d".
        num_stages (int): The number of stages in the PResNet backbone. Defaults to 4.
        return_idx (list[int]): The indices of the stages to return as output. Defaults to [0, 1, 2, 3].
        activation (Callable[..., nn.Module] | None): Activation layer module.
            Defaults to None.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to ``nn.BatchNorm2d``.
        freeze_at (int): The stage at which to freeze the parameters. Defaults to -1.
        pretrained (bool): Whether to load pretrained weights. Defaults to False.
    """

    num_resnet_blocks: ClassVar = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
    }

    donwload_url: ClassVar = {
        18: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth",
        34: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth",
        50: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth",
        101: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth",
    }

    def __init__(
        self,
        depth: int,
        variant: str = "d",
        num_stages: int = 4,
        return_idx: list[int] = [0, 1, 2, 3],  # noqa: B006
        activation: Callable[..., nn.Module] | None = nn.ReLU,
        normalization: Callable[..., nn.Module] = partial(build_norm_layer, nn.BatchNorm2d, layer_name="norm"),
        freeze_at: int = -1,
        pretrained: bool = False,
    ) -> None:
        """Initialize the PResNet backbone."""
        super().__init__()

        block_nums = self.num_resnet_blocks[depth]
        ch_in = 64
        if variant in ["c", "d"]:
            conv_def: list[list[Any]] = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        _name,
                        Conv2dModule(
                            c_in,
                            c_out,
                            k,
                            s,
                            padding=(k - 1) // 2,
                            normalization=build_norm_layer(normalization, num_features=c_out),
                            activation=build_activation_layer(activation),
                        ),
                    )
                    for c_in, c_out, k, s, _name in conv_def
                ],
            ),
        )

        ch_out_list = [64, 128, 256, 512]
        block: nn.Module = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(
                    block,
                    ch_in,
                    ch_out_list[i],
                    block_nums[i],
                    stage_num,
                    activation=activation,
                    variant=variant,
                    normalization=normalization,
                ),
            )
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if pretrained:
            state = torch.hub.load_state_dict_from_url(self.donwload_url[depth])
            self.load_state_dict(state)
            print(f"Load PResNet{depth} state_dict")

    def _freeze_parameters(self, m: nn.Module) -> None:
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        conv1 = self.conv1(x)
        x = nn.functional.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
