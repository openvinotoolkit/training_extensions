# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""High Performance GPU Net(HGNet) Backbone from PaddlePaddle.

Modified from:
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    https://github.com/Peterande/D-FINE
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as f
from torch import Tensor, nn

from getitune.backend.lightning.models.modules.norm import FrozenBatchNorm2d

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_


class LearnableAffineBlock(nn.Module):
    """Learnable affine block.

    Args:
        scale_value (float, optional): scale. Defaults to 1.0.
        bias_value (float, optional): bias. Defaults to 0.0.
    """

    def __init__(
        self,
        scale_value: float = 1.0,
        bias_value: float = 0.0,
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: output tensor.
        """
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    """Convolutional block with batch normalization and activation.

        TODO(Eugene): External LAB is embedded. 'Try'? switching to getitune ConvModule implementation in next PR.

    Args:
        in_channels (int): In channels.
        out_channels (int): Out Channels.
        kernel_size (int): convolution kernel size.
        stride (int, optional): stride. Defaults to 1.
        groups (int, optional): number of conv groups. Defaults to 1.
        use_act (bool, optional): Use ReLU activation. Defaults to True.
        use_lab (bool, optional): Use learnable affine block. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        use_act: bool = True,
        use_lab: bool = False,
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: output tensor.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.lab(x)


class LightConvBNAct(nn.Module):
    """Lightweight convolutional block with batch normalization and activation.

    Args:
    in_chs (int): In channels.
    out_chs (int): Out channels.
    kernel_size (int): convolution kernel size.
    use_lab (bool, optional): Use Learnable Affine Block. Defaults to False.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        use_lab: bool = False,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: output tensor.
        """
        x = self.conv1(x)
        return self.conv2(x)


class HGNetv2StemBlock(nn.Module):
    """HGNetV2 stem block.

    Args:
        in_chs (int): In channels.
        mid_chs (int): Mid channels.
        out_chs (int): Out channels.
        use_lab (bool, optional): Use Learnable Affine Block. Defaults to False.
    """

    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        use_lab: bool = False,
    ):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: output tensor.
        """
        x = self.stem1(x)
        x = f.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = f.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        return self.stem4(x)


class HGBlock(nn.Module):
    """HGNetV2 block.

    Args:
        in_chs (int): In channels.
        mid_chs (int): Mid channels.
        out_chs (int): Out channels.
        layer_num (int): Number of convolutional layers.
        kernel_size (int, optional): kernel size. Defaults to 3.
        residual (bool, optional): Add residual. Defaults to False.
        light_block (bool, optional): Use LightConvBNAct layer. Defaults to False.
        use_lab (bool, optional): User Learnable Affine Block. Defaults to False.
        drop_path (float, optional): Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        layer_num: int,
        kernel_size: int = 3,
        residual: bool = False,
        light_block: bool = False,
        use_lab: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    ),
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    ),
                )

        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        aggregation_squeeze_conv = ConvBNAct(
            total_chs,
            out_chs // 2,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
        )
        aggregation_excitation_conv = ConvBNAct(
            out_chs // 2,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
        )
        self.aggregation = nn.Sequential(
            aggregation_squeeze_conv,
            aggregation_excitation_conv,
        )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: output tensor.
        """
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            return self.drop_path(x) + identity
        return x


class HGStage(nn.Module):
    """HGNetV2 Stage Block.

    Args:
        in_chs (int): In channels.
        mid_chs (int): Mid channels.
        out_chs (int): Out channels.
        block_num (int): Number of blocks.
        layer_num (int): Number of convolutional layers.
        downsample (bool, optional): Downsample. Defaults to True.
        light_block (bool, optional): Use LightConvBNAct layer. Defaults to False.
        kernel_size (int, optional): kernel size. Defaults to 3.
        use_lab (bool, optional): User Learnable Affine Block. Defaults to False.
        drop_path (float, optional): Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        block_num: int,
        layer_num: int,
        downsample: bool = True,
        light_block: bool = False,
        kernel_size: int = 3,
        use_lab: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.downsample = (
            ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
            if downsample
            else nn.Identity()
        )

        blocks_list = [
            HGBlock(
                out_chs if i > 0 else in_chs,
                mid_chs,
                out_chs,
                layer_num,
                residual=i > 0,
                kernel_size=kernel_size,
                light_block=light_block,
                use_lab=use_lab,
                drop_path=drop_path,
            )
            for i in range(block_num)
        ]
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: output tensor.
        """
        x = self.downsample(x)
        return self.blocks(x)


class HGNetv2Module(nn.Module):
    """HGNetV2 Module.

    Args:
        name (str): backbone name (i.e. B0, B2, B4, B5).
        use_lab (bool, optional): User Learnable Affine Block. Defaults to False.
        return_idx (list[int], optional): Feature Maps. Defaults to [1, 2, 3].
        freeze_stem_only (bool, optional): Freeze Stem only. Defaults to True.
        freeze_at (int, optional): Freeze at which stage block. Defaults to 0.
        freeze_norm (bool, optional): Freeze normalization or not. Defaults to True.
    """

    arch_configs: ClassVar[dict[str, dict[str, Any]]] = {
        "B0": {
            "stem_channels": [3, 16, 16],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
        },
        "B2": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
        },
        "B4": {
            "stem_channels": [3, 32, 48],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
        },
        "B5": {
            "stem_channels": [3, 32, 64],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
        },
    }

    def __init__(
        self,
        name: str,
        use_lab: bool = False,
        return_idx: tuple = (1, 2, 3),
        freeze_stem_only: bool = True,
        freeze_at: int = 0,
        freeze_norm: bool = True,
    ) -> None:
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        # Explicit type annotations for static analysis
        stem_channels: list[int] = list(self.arch_configs[name]["stem_channels"])
        stage_config: dict[str, list[int | bool]] = dict(self.arch_configs[name]["stage_config"])

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.stem = HGNetv2StemBlock(
            in_chs=stem_channels[0],
            mid_chs=stem_channels[1],
            out_chs=stem_channels[2],
            use_lab=use_lab,
        )

        # stages
        self.stages = nn.ModuleList()
        for stage in stage_config.values():
            (
                in_channels,
                mid_channels,
                out_channels,
                block_num,
                downsample,
                light_block,
                kernel_size,
                layer_num,
            ) = stage
            self.stages.append(
                HGStage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                ),
            )

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

    def _freeze_norm(self, m: nn.Module) -> nn.Module:
        """Freeze normalization layers.

        Args:
            m (nn.Module): Normalization module.

        Returns:
            nn.Module: Freezed normalization module.
        """
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module) -> None:
        """Freeze module parameters.

        Args:
            m (nn.Module): Module to freeze.
        """
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            list[Tensor]: Output tensor.
        """
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


class HGNetv2:
    """HGNetV2 backbone."""

    backbone_cfg: ClassVar[dict[str, Any]] = {
        "dfine_hgnetv2_n": {
            "name": "B0",
            "return_idx": [2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
            "freeze_stem_only": True,
        },
        "dfine_hgnetv2_s": {
            "name": "B0",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
        },
        "dfine_hgnetv2_m": {
            "name": "B2",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
        },
        "dfine_hgnetv2_l": {
            "name": "B4",
            "return_idx": [1, 2, 3],
            "freeze_at": 0,
            "freeze_norm": True,
            "freeze_stem_only": True,
        },
        "dfine_hgnetv2_x": {
            "name": "B5",
            "return_idx": [1, 2, 3],
            "freeze_at": 0,
            "freeze_norm": True,
            "freeze_stem_only": True,
        },
        "deim_dfine_hgnetv2_n": {
            "name": "B0",
            "return_idx": [2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
            "freeze_stem_only": True,
        },
        "deim_dfine_hgnetv2_s": {
            "name": "B0",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
        },
        "deim_dfine_hgnetv2_m": {
            "name": "B2",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
        },
        "deim_dfine_hgnetv2_l": {
            "name": "B4",
            "return_idx": [1, 2, 3],
            "freeze_at": 0,
            "freeze_norm": True,
            "freeze_stem_only": True,
        },
        "deim_dfine_hgnetv2_x": {
            "name": "B5",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "freeze_stem_only": True,
        },
    }

    def __new__(cls, model_name: str) -> HGNetv2Module:
        """Create HGNetV2 backbone.

        Args:
            model_name (str): Model name.

        Returns:
            HGNetv2Module: HGNetV2 backbone.
        """
        return HGNetv2Module(**cls.backbone_cfg[model_name])
