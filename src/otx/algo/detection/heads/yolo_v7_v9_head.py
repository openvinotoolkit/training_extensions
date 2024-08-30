# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Head implementation of YOLOv7 and YOLOv9."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor, nn

from otx.algo.detection.backbones.yolo_v7_v9_backbone import (
    AConv,
    Conv,
    RepNCSPELAN,
    build_layer,
    module_list_forward,
)
from otx.algo.detection.necks.yolo_v7_v9_neck import SPPELAN, Concat, UpSample
from otx.core.data.entity.detection import DetBatchDataEntity


def round_up(x: int | Tensor, div: int = 1) -> int | Tensor:
    """Rounds up `x` to the bigger-nearest multiple of `div`."""
    return x + (-x % div)


class Anchor2Vec(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        reverse_reg = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1, 1)
        self.anc2vec = nn.Conv3d(in_channels=reg_max, out_channels=1, kernel_size=1, bias=False)
        self.anc2vec.weight = nn.Parameter(reverse_reg, requires_grad=False)

    def forward(self, anchor_x: Tensor) -> Tensor:
        anchor_x = rearrange(anchor_x, "B (P R) h w -> B R P h w", P=4)
        vector_x = anchor_x.softmax(dim=1)
        vector_x = self.anc2vec(vector_x)[:, 0]
        return anchor_x, vector_x


class ImplicitA(nn.Module):
    """Implement YOLOR - implicit knowledge(Add), paper: https://arxiv.org/abs/2105.04206"""

    def __init__(self, channel: int, mean: float = 0.0, std: float = 0.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std

        self.implicit = nn.Parameter(torch.empty(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.implicit + x


class ImplicitM(nn.Module):
    """Implement YOLOR - implicit knowledge(multiply), paper: https://arxiv.org/abs/2105.04206"""

    def __init__(self, channel: int, mean: float = 1.0, std: float = 0.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std

        self.implicit = nn.Parameter(torch.empty(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.implicit * x


class Detection(nn.Module):
    """A single YOLO Detection head for detection models"""

    def __init__(self, in_channels: tuple[int], num_classes: int, *, reg_max: int = 16, use_group: bool = True):
        super().__init__()

        groups = 4 if use_group else 1
        anchor_channels = 4 * reg_max

        first_neck, in_channels = in_channels
        anchor_neck = max(round_up(first_neck // 4, groups), anchor_channels, reg_max)
        class_neck = max(first_neck, min(num_classes * 2, 128))

        self.anchor_conv = nn.Sequential(
            Conv(in_channels, anchor_neck, 3),
            Conv(anchor_neck, anchor_neck, 3, groups=groups),
            nn.Conv2d(anchor_neck, anchor_channels, 1, groups=groups),
        )
        self.class_conv = nn.Sequential(
            Conv(in_channels, class_neck, 3),
            Conv(class_neck, class_neck, 3),
            nn.Conv2d(class_neck, num_classes, 1),
        )

        self.anc2vec = Anchor2Vec(reg_max=reg_max)

        self.anchor_conv[-1].bias.data.fill_(1.0)
        self.class_conv[-1].bias.data.fill_(-10)  # TODO: math.log(5 * 4 ** idx / 80 ** 3)

    def forward(self, x: Tensor) -> tuple[Tensor]:
        anchor_x = self.anchor_conv(x)
        class_x = self.class_conv(x)
        anchor_x, vector_x = self.anc2vec(anchor_x)
        return class_x, anchor_x, vector_x


class IDetection(nn.Module):
    def __init__(self, in_channels: tuple[int], num_classes: int, *args, anchor_num: int = 3, **kwargs):
        super().__init__()

        if isinstance(in_channels, tuple):
            in_channels = in_channels[1]

        out_channel = num_classes + 5
        out_channels = out_channel * anchor_num
        self.head_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.implicit_a = ImplicitA(in_channels)
        self.implicit_m = ImplicitM(out_channels)

    def forward(self, x):
        x = self.implicit_a(x)
        x = self.head_conv(x)
        x = self.implicit_m(x)

        return x


class MultiheadDetection(nn.Module):
    """Mutlihead Detection module for Dual detect or Triple detect"""

    def __init__(self, in_channels: list[int], num_classes: int, **head_kwargs):
        super().__init__()
        DetectionHead = Detection

        if head_kwargs.pop("version", None) == "v7":
            DetectionHead = IDetection

        self.heads = nn.ModuleList(
            [DetectionHead((in_channels[0], in_channel), num_classes, **head_kwargs) for in_channel in in_channels],
        )

    def forward(self, x_list: list[Tensor]) -> list[Tensor]:
        return [head(x) for x, head in zip(x_list, self.heads)]


def patch_bbox_head_loss(
    self: nn.ModuleList,
    x: Tensor | dict[str, Tensor],
    entity: DetBatchDataEntity,
    *args,
    **kwargs,
) -> dict[str, Tensor]:
    outputs = self(x)

    # pad and concatenate labels and bboxes
    max_len = max(len(b) for b in entity.bboxes)
    padded_labels = [
        nn.functional.pad(label.unsqueeze(1), (0, 0, 0, max_len - len(label)), value=-1) for label in entity.labels
    ]
    padded_bboxes = [nn.functional.pad(box, (0, 0, 0, max_len - len(box)), value=0) for box in entity.bboxes]
    merged_padded_labels_bboxes = torch.stack(
        [torch.cat((label, box), dim=1) for label, box in zip(padded_labels, padded_bboxes)], dim=0
    )
    return {"main_predicts": outputs["Main"], "aux_predicts": outputs["AUX"], "targets": merged_padded_labels_bboxes}


class YOLOv9Head(nn.Module):
    """Head for YOLOv9.

    TODO (sungchul): change head name
    """

    def __new__(cls, model_name: str, num_classes: int) -> nn.ModuleList:
        nn.ModuleList.forward = module_list_forward
        nn.ModuleList.loss = patch_bbox_head_loss
        nn.ModuleList.predict = module_list_forward
        if model_name == "yolov9-s":
            return nn.ModuleList(
                [
                    build_layer({"module": UpSample(scale_factor=2, mode="nearest")}),
                    build_layer({"module": Concat(), "source": [-1, "B3"]}),
                    build_layer(
                        {"module": RepNCSPELAN(320, 128, part_channels=128, csp_args={"repeat_num": 3}), "tags": "P3"},
                    ),
                    build_layer({"module": AConv(128, 96)}),
                    build_layer({"module": Concat(), "source": [-1, "N4"]}),
                    build_layer(
                        {"module": RepNCSPELAN(288, 192, part_channels=192, csp_args={"repeat_num": 3}), "tags": "P4"},
                    ),
                    build_layer({"module": AConv(192, 128)}),
                    build_layer({"module": Concat(), "source": [-1, "N3"]}),
                    build_layer(
                        {"module": RepNCSPELAN(384, 256, part_channels=256, csp_args={"repeat_num": 3}), "tags": "P5"},
                    ),
                    build_layer(
                        {
                            "module": MultiheadDetection([128, 192, 256], num_classes),
                            "source": ["P3", "P4", "P5"],
                            "tags": "Main",
                            "output": True,
                        },
                    ),
                    build_layer({"module": SPPELAN(256, 256), "source": "B5", "tags": "A5"}),
                    build_layer({"module": UpSample(scale_factor=2, mode="nearest")}),
                    build_layer({"module": Concat(), "source": [-1, "B4"]}),
                    build_layer(
                        {"module": RepNCSPELAN(448, 192, part_channels=192, csp_args={"repeat_num": 3}), "tags": "A4"},
                    ),
                    build_layer({"module": UpSample(scale_factor=2, mode="nearest")}),
                    build_layer({"module": Concat(), "source": [-1, "B3"]}),
                    build_layer(
                        {"module": RepNCSPELAN(320, 128, part_channels=128, csp_args={"repeat_num": 3}), "tags": "A3"},
                    ),
                    build_layer(
                        {
                            "module": MultiheadDetection([128, 192, 256], num_classes),
                            "source": ["A3", "A4", "A5"],
                            "tags": "AUX",
                            "output": True,
                        },
                    ),
                ],
            )

        # if model_name == "v9-m":
        #     return nn.ModuleList(
        #         [
        #             RepNCSPELAN(360, 240, part_channels=240),
        #             AConv(240, 184),
        #             Concat(),
        #             RepNCSPELAN(184, 360, part_channels=360),
        #             AConv(360, 240),
        #             Concat(),
        #             RepNCSPELAN(240, 480, part_channels=480),
        #         ]
        #     )

        # if model_name == "v9-c":
        #     return nn.ModuleList(
        #         [
        #             RepNCSPELAN(512, 256, part_channels=256),
        #             ADown(256, 256),
        #             Concat(),
        #             RepNCSPELAN(256, 512, part_channels=512),
        #             ADown(512, 512),
        #             Concat(),
        #             RepNCSPELAN(512, 512, part_channels=512),
        #         ]
        #     )

        msg = f"Unknown model_name: {model_name}"
        raise ValueError(msg)

    def forward(self, outputs: dict[str, Tensor]) -> Tensor:
        final_outputs: dict[str, Tensor] = {}
        x = outputs[-1]
        for module in self:
            if isinstance(module, tuple):
                if isinstance(source := module[0], str):
                    x = module[1](x)
                    outputs[source] = x
                elif isinstance(source := module[0], list):
                    # for Concat
                    x = module[1]([outputs[key] for key in source])
            else:
                x = module(x)
            outputs[-1] = x
        return outputs
