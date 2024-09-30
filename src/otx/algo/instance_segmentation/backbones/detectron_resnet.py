# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Implementation modified from Detectron2 ResNet.

Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/resnet.py
"""


from __future__ import annotations

import numpy as np
import torch.nn.functional as f
from torch import Tensor, nn

from otx.algo.instance_segmentation.layers.batch_norm import CNNBlockBase, get_norm
from otx.algo.instance_segmentation.utils.utils import Conv2d, ShapeSpec, c2_msra_fill

__all__ = [
    "BottleneckBlock",
    "BasicStem",
    "ResNet",
    "build_resnet_backbone",
]


class BottleneckBlock(CNNBlockBase):
    """The standard bottleneck residual block used by ResNet-50, 101 and 152."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        stride: int = 1,
        num_groups: int = 1,
        norm: str = "BN",
        stride_in_1x1: bool = False,
        dilation: int = 1,
    ) -> None:
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None  # type: ignore[assignment]

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                c2_msra_fill(layer)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        out = self.conv1(x)
        out = f.relu_(out)

        out = self.conv2(out)
        out = f.relu_(out)

        out = self.conv3(out)

        shortcut = self.shortcut(x) if self.shortcut is not None else x

        out += shortcut
        return f.relu_(out)


class BasicStem(CNNBlockBase):
    """The standard ResNet stem (layers before the first residual block), with a conv, relu and max_pool."""

    def __init__(self, in_channels: int = 3, out_channels: int = 64, norm: str = "BN") -> None:
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        c2_msra_fill(self.conv1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = f.relu_(x)
        return f.max_pool2d(x, kernel_size=3, stride=2, padding=1)


class ResNet(nn.Module):
    """Implement :paper:`ResNet`."""

    def __init__(
        self,
        stem: nn.Module,
        stages: list[list[CNNBlockBase]],
        out_features: tuple[str, ...],
        freeze_at: int = 0,
    ) -> None:
        super().__init__()
        self.stem = stem

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names, self.stages = [], []

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                [{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features],
            )
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks]),
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

        # Make it static for scripting
        self.stage_names = tuple(self.stage_names)  # type: ignore [assignment]

        self._out_features = out_features
        self.freeze(freeze_at)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass."""
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages, strict=True):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        return outputs

    def freeze(self, freeze_at: int = 0) -> nn.Module:
        """Freeze the first several stages of the ResNet. Commonly used in fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(
        block_class: nn.Module,
        num_blocks: int,
        *,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> list[CNNBlockBase]:
        """Create a list of blocks of the same type that forms one ResNet stage.

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.

        Returns:
            list[CNNBlockBase]: a list of block module.

        Examples:
        ::
            stage = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )

        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    newk = k[: -len("_per_block")]
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs),
            )
            in_channels = out_channels
        return blocks

    def output_shape(self) -> dict[str, ShapeSpec]:
        """Returns output shapes for each stage."""
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


def build_resnet_backbone(
    norm: str,
    stem_out_channels: int,
    input_shape: ShapeSpec,
    freeze_at: int,
    out_features: tuple[str, ...],
    depth: int,
    num_groups: int,
    width_per_group: int,
    in_channels: int,
    out_channels: int,
    stride_in_1x1: bool,
    res5_dilation: int = 1,
) -> ResNet:
    """Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=stem_out_channels,
        norm=norm,
    )

    bottleneck_channels = width_per_group * num_groups
    num_blocks_per_stage = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    stages = []

    for idx, stage_idx in enumerate(range(2, 6)):
        # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        stage_kargs["bottleneck_channels"] = bottleneck_channels
        stage_kargs["stride_in_1x1"] = stride_in_1x1
        stage_kargs["dilation"] = dilation
        stage_kargs["num_groups"] = num_groups
        stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)  # type: ignore[arg-type]
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)
