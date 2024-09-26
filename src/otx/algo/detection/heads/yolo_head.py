# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Head implementation of YOLOv7 and YOLOv9.

Reference : https://github.com/WongKinYiu/YOLO
"""

from __future__ import annotations  # noqa: I001

from typing import Any, ClassVar, NoReturn

import torch
from einops import rearrange
from torch import Tensor, nn
from torchvision.ops import batched_nms

from otx.algo.common.utils.nms import multiclass_nms
from otx.algo.detection.heads.base_head import BaseDenseHead
from otx.algo.detection.layers import AConv, ADown, Concat, SPPELAN, RepNCSPELAN
from otx.algo.detection.utils.utils import round_up, set_info_into_instance, auto_pad
from otx.algo.modules import Conv2dModule
from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.base import OTXBatchDataEntity
from otx.core.data.entity.detection import DetBatchDataEntity


class Anchor2Vec(nn.Module):
    """Convert anchor tensor to vector tensor.

    Args:
        reg_max (int): Maximum number of anchor regions.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        reverse_reg = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1, 1)
        self.anc2vec = nn.Conv3d(in_channels=reg_max, out_channels=1, kernel_size=1, bias=False)
        self.anc2vec.weight = nn.Parameter(reverse_reg, requires_grad=False)

    def forward(self, anchor_x: Tensor) -> Tensor:
        """Forward function."""
        anchor_x = rearrange(anchor_x, "B (P R) h w -> B R P h w", P=4)
        vector_x = anchor_x.softmax(dim=1)
        vector_x = self.anc2vec(vector_x)[:, 0]
        return anchor_x, vector_x


class CBLinear(nn.Module):
    """Convolutional block that outputs multiple feature maps split along the channel dimension.

    Args:
        in_channels (int): Number of input channels.
        out_channels (list[int]): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
    """

    def __init__(self, in_channels: int, out_channels: list[int], kernel_size: int = 1, **kwargs) -> None:
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.conv = nn.Conv2d(in_channels, sum(out_channels), kernel_size, **kwargs)
        self.out_channels = list(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x = self.conv(x)
        return x.split(self.out_channels, dim=1)


class CBFuse(nn.Module):
    """Fuse the feature maps from the previous layer with the feature maps from the current layer.

    Args:
        index (list[int]): Index of the feature maps from the previous layer.
        mode (str): Mode of the interpolation operation.
    """

    def __init__(self, index: list[int], mode: str = "nearest") -> None:
        super().__init__()
        self.idx = index
        self.mode = mode

    def forward(self, x_list: list[tuple[Tensor, ...] | Tensor]) -> Tensor:
        """Forward function."""
        target: Tensor = x_list[-1]
        target_size = target.shape[2:]  # Batch, Channel, H, W

        res = [
            nn.functional.interpolate(x[pick_id], size=target_size, mode=self.mode)
            for pick_id, x in zip(self.idx, x_list)
        ]
        return torch.stack([*res, target]).sum(dim=0)


class ImplicitA(nn.Module):
    """Implement YOLOR - implicit knowledge(Add).

    paper: https://arxiv.org/abs/2105.04206

    Args:
        channel (int): Number of input channels.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
    """

    def __init__(self, channel: int, mean: float = 0.0, std: float = 0.02) -> None:
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std

        self.implicit = nn.Parameter(torch.empty(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=self.std)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        return self.implicit + x


class ImplicitM(nn.Module):
    """Implement YOLOR - implicit knowledge(multiply).

    paper: https://arxiv.org/abs/2105.04206

    Args:
        channel (int): Number of input channels.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
    """

    def __init__(self, channel: int, mean: float = 1.0, std: float = 0.02) -> None:
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std

        self.implicit = nn.Parameter(torch.empty(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        return self.implicit * x


class SingleHeadDetectionforYOLOv9(nn.Module):
    """A single YOLO Detection head for YOLOv9 detection models.

    Args:
        in_channels (tuple[int, int]): Number of input channels.
        num_classes (int): Number of classes.
        reg_max (int): Maximum number of anchor regions.
        use_group (bool): Whether to use group convolution.
    """

    def __init__(
        self,
        in_channels: tuple[int, int],
        num_classes: int,
        *,
        reg_max: int = 16,
        use_group: bool = True,
    ) -> None:
        super().__init__()

        groups = 4 if use_group else 1
        anchor_channels = 4 * reg_max

        first_neck, first_channels = in_channels
        anchor_neck = max(round_up(first_neck // 4, groups), anchor_channels, reg_max)
        class_neck = max(first_neck, min(num_classes * 2, 128))

        self.anchor_conv = nn.Sequential(
            Conv2dModule(
                first_channels,
                anchor_neck,
                3,
                padding=auto_pad(3),
                normalization=nn.BatchNorm2d(anchor_neck, eps=1e-3, momentum=3e-2),
                activation=nn.SiLU(inplace=True),
            ),
            Conv2dModule(
                anchor_neck,
                anchor_neck,
                3,
                padding=auto_pad(3),
                groups=groups,
                normalization=nn.BatchNorm2d(anchor_neck, eps=1e-3, momentum=3e-2),
                activation=nn.SiLU(inplace=True),
            ),
            nn.Conv2d(anchor_neck, anchor_channels, 1, groups=groups),
        )
        self.class_conv = nn.Sequential(
            Conv2dModule(
                first_channels,
                class_neck,
                3,
                padding=auto_pad(3),
                normalization=nn.BatchNorm2d(class_neck, eps=1e-3, momentum=3e-2),
                activation=nn.SiLU(inplace=True),
            ),
            Conv2dModule(
                class_neck,
                class_neck,
                3,
                padding=auto_pad(3),
                normalization=nn.BatchNorm2d(class_neck, eps=1e-3, momentum=3e-2),
                activation=nn.SiLU(inplace=True),
            ),
            nn.Conv2d(class_neck, num_classes, 1),
        )

        self.anc2vec = Anchor2Vec(reg_max=reg_max)

        self.anchor_conv[-1].bias.data.fill_(1.0)
        self.class_conv[-1].bias.data.fill_(-10)  # TODO (author): math.log(5 * 4 ** idx / 80 ** 3)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward function."""
        anchor_x = self.anchor_conv(x)
        class_x = self.class_conv(x)
        anchor_x, vector_x = self.anc2vec(anchor_x)
        return class_x, anchor_x, vector_x


class SingleHeadDetectionforYOLOv7(nn.Module):
    """A single YOLO Detection head for YOLOv7 detection models.

    Args:
        in_channels (int | tuple[int, int]): Number of input channels.
        num_classes (int): Number of classes.
        anchor_num (int): Number of anchors. Default is 3.
    """

    def __init__(
        self,
        in_channels: int | tuple[int, int],
        num_classes: int,
        *args,
        anchor_num: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(in_channels, tuple):
            in_channels = in_channels[1]

        out_channel = num_classes + 5
        out_channels = out_channel * anchor_num
        self.head_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.implicit_a = ImplicitA(in_channels)
        self.implicit_m = ImplicitM(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x = self.implicit_a(x)
        x = self.head_conv(x)
        return self.implicit_m(x)


class MultiheadDetection(nn.Module):
    """Mutlihead Detection module for Dual detect or Triple detect.

    Args:
        in_channels (list[int]): Number of input channels.
        num_classes (int): Number of classes.
    """

    def __init__(self, in_channels: list[int], num_classes: int, **head_kwargs) -> None:
        super().__init__()
        single_head_detection: nn.Module = (
            SingleHeadDetectionforYOLOv7 if head_kwargs.pop("version", None) == "v7" else SingleHeadDetectionforYOLOv9
        )

        self.heads = nn.ModuleList(
            [
                single_head_detection((in_channels[0], in_channel), num_classes, **head_kwargs)
                for in_channel in in_channels
            ],
        )

    def forward(self, x_list: list[Tensor]) -> list[Tensor]:
        """Forward function."""
        return [head(x) for x, head in zip(x_list, self.heads)]


class YOLOHeadModule(BaseDenseHead):
    """Head for YOLOv7 and v9.

    Args:
        num_classes (int): Number of classes.
        csp_channels (list[list[int]]): List of channels for CSP blocks.
        concat_sources (list[str | int]): List of sources to concatenate.
        aconv_channels (list[list[int]], optional): List of channels for AConv. Defaults to None.
        adown_channels (list[list[int]], optional): List of channels for ADown. Defaults to None.
        pre_upsample_concat_cfg (dict[str, Any], optional): Configuration for pre-upsampling. Defaults to None.
        csp_args (dict[str, Any], optional): Arguments for CSP blocks. Defaults to None.
        aux_cfg (dict[str, Any], optional): Configuration for auxiliary head. Defaults to None.
        with_nms (bool, optional): Whether to use NMS. Defaults to True.
        min_confidence (float, optional): Minimum confidence for NMS. Defaults to 0.1.
        min_iou (float, optional): Minimum IoU for NMS. Defaults to 0.65.
    """

    def __init__(
        self,
        num_classes: int,
        csp_channels: list[list[int]],
        concat_sources: list[str | int],
        aconv_channels: list[list[int]] | None = None,
        adown_channels: list[list[int]] | None = None,
        pre_upsample_concat_cfg: dict[str, Any] | None = None,
        csp_args: dict[str, Any] | None = None,
        aux_cfg: dict[str, Any] | None = None,
        with_nms: bool = True,
        min_confidence: float = 0.1,
        min_iou: float = 0.65,
    ) -> None:
        if len(csp_channels) - 1 != len(concat_sources):
            msg = (
                f"len(csp_channels) - 1 ({len(csp_channels) - 1}) "
                f"and len(concat_sources) ({len(concat_sources)}) should be the same."
            )
            raise ValueError(msg)

        super().__init__()
        self.num_classes = num_classes
        self.csp_channels = csp_channels
        self.aconv_channels = aconv_channels
        self.concat_sources = concat_sources
        self.pre_upsample_concat_cfg = pre_upsample_concat_cfg
        self.csp_args = csp_args or {}
        self.aux_cfg = aux_cfg
        self.with_nms = with_nms
        self.min_confidence = min_confidence
        self.min_iou = min_iou

        self.module = nn.ModuleList()
        if pre_upsample_concat_cfg:
            # for yolov9_s
            self.module.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.module.append(
                set_info_into_instance({"module": Concat(), "source": pre_upsample_concat_cfg.get("source")}),
            )

        output_channels: list[int] = []
        self.module.append(
            set_info_into_instance(
                {
                    "module": RepNCSPELAN(
                        csp_channels[0][0],
                        csp_channels[0][1],
                        part_channels=csp_channels[0][2],
                        csp_args=self.csp_args,
                    ),
                    "tags": "P3",
                },
            ),
        )
        output_channels.append(csp_channels[0][1])

        aconv_adown_channels = aconv_channels or adown_channels
        if aconv_adown_channels is None:
            msg = "Only one of aconv_channels or adown_channels should be provided."
            raise ValueError(msg)
        aconv_adown_object = AConv if aconv_channels else ADown
        for idx, (csp_channel, aconv_adown_channel, concat_source) in enumerate(
            zip(csp_channels[1:], aconv_adown_channels, concat_sources),
            start=4,
        ):
            self.module.append(aconv_adown_object(aconv_adown_channel[0], aconv_adown_channel[1]))
            self.module.append(set_info_into_instance({"module": Concat(), "source": concat_source}))
            self.module.append(
                set_info_into_instance(
                    {
                        "module": RepNCSPELAN(
                            csp_channel[0],
                            csp_channel[1],
                            part_channels=csp_channel[2],
                            csp_args=self.csp_args,
                        ),
                        "tags": f"P{idx}",
                    },
                ),
            )
            output_channels.append(csp_channel[1])

        self.module.append(
            set_info_into_instance(
                {
                    "module": MultiheadDetection(output_channels, num_classes),
                    "source": ["P3", "P4", "P5"],
                    "tags": "Main",
                    "output": True,
                },
            ),
        )

        if aux_cfg:
            aux_output_channels: list[int] = []
            if sppelan_channels := aux_cfg.get("sppelan_channels", None):
                # for yolov9_s
                self.module.append(
                    set_info_into_instance(
                        {"module": SPPELAN(sppelan_channels[0], sppelan_channels[1]), "source": "B5", "tags": "A5"},
                    ),
                )
                aux_output_channels.append(sppelan_channels[1])
                for idx, csp_channel in enumerate(aux_cfg.get("csp_channels", [])):
                    self.module.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    self.module.append(set_info_into_instance({"module": Concat(), "source": [-1, f"B{4-idx}"]}))
                    self.module.append(
                        set_info_into_instance(
                            {
                                "module": RepNCSPELAN(
                                    csp_channel[0],
                                    csp_channel[1],
                                    part_channels=csp_channel[2],
                                    csp_args=self.csp_args,
                                ),
                                "tags": f"A{4-idx}",
                            },
                        ),
                    )
                    aux_output_channels.append(csp_channel[1])
                aux_output_channels = aux_output_channels[::-1]  # reverse channels

            elif cblinear_channels := aux_cfg.get("cblinear_channels", None):
                # for yolov9_m, c
                for idx, cblinear_channel in enumerate(cblinear_channels, start=3):
                    self.module.append(
                        set_info_into_instance(
                            {
                                "module": CBLinear(cblinear_channel[0], cblinear_channel[1]),
                                "source": f"B{idx}",
                                "tags": f"R{idx}",
                            },
                        ),
                    )

                aux_aconv_adown_channels = aux_cfg.get("aconv_channels", None) or aux_cfg.get("adown_channels", None)
                if aux_aconv_adown_channels is None:
                    msg = "Only one of aconv_channels or adown_channels should be provided."
                    raise ValueError(msg)

                aux_aconv_adown_object = AConv if aconv_channels else ADown
                for idx, (csp_channel, aux_aconv_adown_channel, cbfuse_index, cbfuse_source) in enumerate(
                    zip(
                        aux_cfg.get("csp_channels", []),
                        aux_aconv_adown_channels,
                        aux_cfg.get("cbfuse_indices", []),
                        aux_cfg.get("cbfuse_sources", []),
                    ),
                ):
                    if idx == 0 and len(aux_aconv_adown_channel) == 0 and len(cbfuse_index) == 0:
                        conv_channels: list[list[int]] = aux_cfg.get("conv_channels")  # type: ignore[assignment]
                        self.module.append(
                            set_info_into_instance(
                                {
                                    "module": Conv2dModule(
                                        conv_channels[0][0],
                                        conv_channels[0][1],
                                        3,
                                        stride=2,
                                        padding=auto_pad(3),
                                        normalization=nn.BatchNorm2d(conv_channels[0][1], eps=1e-3, momentum=3e-2),
                                        activation=nn.SiLU(inplace=True),
                                    ),
                                    "source": 0,
                                },
                            ),
                        )
                        self.module.append(
                            Conv2dModule(
                                conv_channels[1][0],
                                conv_channels[1][1],
                                3,
                                stride=2,
                                padding=auto_pad(3),
                                normalization=nn.BatchNorm2d(conv_channels[1][1], eps=1e-3, momentum=3e-2),
                                activation=nn.SiLU(inplace=True),
                            ),
                        )
                        self.module.append(RepNCSPELAN(csp_channel[0], csp_channel[1], part_channels=csp_channel[2]))
                    else:
                        self.module.append(
                            aux_aconv_adown_object(aux_aconv_adown_channel[0], aux_aconv_adown_channel[1]),
                        )
                        self.module.append(
                            set_info_into_instance({"module": CBFuse(cbfuse_index), "source": cbfuse_source}),
                        )
                        self.module.append(
                            set_info_into_instance(
                                {
                                    "module": RepNCSPELAN(csp_channel[0], csp_channel[1], part_channels=csp_channel[2]),
                                    "tags": f"A{idx+2}",
                                },
                            ),
                        )
                        aux_output_channels.append(csp_channel[1])

            self.module.append(
                set_info_into_instance(
                    {
                        "module": MultiheadDetection(aux_output_channels, num_classes),
                        "source": ["A3", "A4", "A5"],
                        "tags": "AUX",
                        "output": True,
                    },
                ),
            )

    @property
    def is_aux(self) -> bool:
        """Check if the head has an auxiliary head."""
        return bool(self.aux_cfg)

    def forward(self, outputs: dict[int | str, Tensor], *args, **kwargs) -> tuple[Tensor, None] | tuple[Tensor, Tensor]:
        """Forward function."""
        for layer in self.module:
            if hasattr(layer, "source") and isinstance(layer.source, list):
                model_input = [outputs[idx] for idx in layer.source]
            else:
                model_input = outputs[getattr(layer, "source", -1)]  # type: ignore[arg-type]
            x = layer(model_input)
            outputs[-1] = x
            if hasattr(layer, "tags"):
                outputs[layer.tags] = x

        if self.is_aux:
            return outputs["Main"], outputs["AUX"]
        return outputs["Main"], None

    def prepare_loss_inputs(self, x: tuple[Tensor], entity: DetBatchDataEntity) -> dict:
        """Perform forward propagation and loss calculation of the detection head.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            entity (DetBatchDataEntity): Entity from OTX dataset.

        Returns:
            dict: A dictionary of loss components.
        """
        main_preds, aux_preds = self(x)

        padded_bboxes, padded_labels = self.pad_bbox_labels(entity.bboxes, entity.labels)
        merged_padded_labels_bboxes = torch.cat((padded_labels, padded_bboxes), dim=-1)
        return {
            "main_preds": main_preds,
            "aux_preds": aux_preds,
            "targets": merged_padded_labels_bboxes,
        }

    def loss_by_feat(self, *args, **kwargs) -> NoReturn:
        """Calculate the loss based on the features extracted by the detection head."""
        raise NotImplementedError

    def predict(
        self,
        x: tuple[Tensor],
        entity: OTXBatchDataEntity,
        rescale: bool = False,
    ) -> list[InstanceData]:
        """Perform forward propagation of the detection head and predict detection results.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            entity (OTXBatchDataEntity): Entity from OTX dataset.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[InstanceData]: Detection results of each image
            after the post process.
        """
        main_preds, _ = self(x)

        pred_bboxes: Tensor | list[Tensor]
        pred_scores: Tensor | list[Tensor]
        pred_labels: Tensor | list[Tensor]

        prediction = self.vec2box(main_preds)
        pred_classes, _, pred_bboxes = prediction[:3]
        pred_scores = pred_classes.sigmoid() * (prediction[3] if len(prediction) == 4 else 1)

        # TODO (sungchul): use otx modules
        pred_scores, pred_labels = pred_scores.max(dim=-1, keepdim=True)
        if rescale:
            # rescale
            scale_factors = [img_info.scale_factor[::-1] for img_info in entity.imgs_info]  # type: ignore[index]
            pred_bboxes /= pred_bboxes.new_tensor(scale_factors).repeat((1, 2)).unsqueeze(1)

        if self.with_nms and pred_bboxes.numel():
            # filter class by confidence
            valid_mask = pred_scores > self.min_confidence
            valid_labels = pred_labels[valid_mask].float()
            valid_scores = pred_scores[valid_mask].float()
            valid_bboxes = pred_bboxes[valid_mask.repeat(1, 1, 4)].view(-1, 4)

            # nms
            batch_idx, *_ = torch.where(valid_mask)
            nms_idx = batched_nms(valid_bboxes, valid_scores, valid_labels, self.min_iou)

            def filter_predictions() -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
                _pred_bboxes = []
                _pred_scores = []
                _pred_labels = []
                for idx in range(pred_classes.size(0)):
                    instance_idx = nms_idx[idx == batch_idx[nms_idx]]
                    _pred_bboxes.append(valid_bboxes[instance_idx])
                    _pred_scores.append(valid_scores[instance_idx])
                    _pred_labels.append(valid_labels[instance_idx])
                return _pred_bboxes, _pred_scores, _pred_labels

            pred_bboxes, pred_scores, pred_labels = filter_predictions()

        return [
            InstanceData(
                bboxes=pred_bboxes[idx],
                scores=pred_scores[idx],
                labels=pred_labels[idx].type(torch.long),
            )
            for idx in range(pred_classes.size(0))
        ]

    def export(
        self,
        x: tuple[Tensor],
        batch_img_metas: list[dict],
        rescale: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Perform forward propagation of the detection head and predict detection results.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream network, each is a 4D-tensor.
            batch_data_samples (list[dict]): The Data Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
                Detection results of each image after the post process.
        """
        main_preds, _ = self(x)

        prediction = self.vec2box(main_preds)
        pred_class, _, pred_bbox = prediction[:3]
        pred_conf = prediction[3] if len(prediction) == 4 else None

        scores = pred_class.sigmoid() * (1 if pred_conf is None else pred_conf)

        return multiclass_nms(
            pred_bbox,
            scores,
            max_output_boxes_per_class=200,
            iou_threshold=self.min_iou,
            score_threshold=self.min_confidence,
            pre_top_k=5000,
            keep_top_k=100,
        )

    def pad_bbox_labels(self, bboxes: list[Tensor], labels: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Pad bounding boxes and labels to the same length."""
        max_len = max(b.shape[0] for b in bboxes)
        padded_labels = torch.stack(
            [nn.functional.pad(label.unsqueeze(1), (0, 0, 0, max_len - label.shape[0]), value=-1) for label in labels],
            dim=0,
        )
        padded_bboxes = torch.stack(
            [nn.functional.pad(box, (0, 0, 0, max_len - box.shape[0]), value=0) for box in bboxes],
            dim=0,
        )
        return padded_bboxes, padded_labels


class YOLOHead:
    """YOLOHead factory for detection."""

    YOLOHEAD_CFG: ClassVar[dict[str, Any]] = {
        "yolov9_s": {
            "csp_channels": [[320, 128, 128], [288, 192, 192], [384, 256, 256]],
            "aconv_channels": [[128, 96], [192, 128]],
            "concat_sources": [[-1, "N4"], [-1, "N3"]],
            "pre_upsample_concat_cfg": {"source": [-1, "B3"]},
            "csp_args": {"repeat_num": 3},
            "aux_cfg": {
                "sppelan_channels": [256, 256],
                "csp_channels": [[448, 192, 192], [320, 128, 128]],
            },
        },
        "yolov9_m": {
            "csp_channels": [[600, 240, 240], [544, 360, 360], [720, 480, 480]],
            "aconv_channels": [[240, 184], [360, 240]],
            "concat_sources": [[-1, "N4"], [-1, "N3"]],
            "aux_cfg": {
                "cblinear_channels": [[240, [240]], [360, [240, 360]], [480, [240, 360, 480]]],
                "csp_channels": [[64, 128, 128], [240, 240, 240], [360, 360, 360], [480, 480, 480]],
                "conv_channels": [[3, 32], [32, 64]],
                "aconv_channels": [[], [128, 240], [240, 360], [360, 480]],
                "cbfuse_indices": [[], [0, 0, 0], [1, 1], [2]],
                "cbfuse_sources": [[], ["R3", "R4", "R5", -1], ["R4", "R5", -1], ["R5", -1]],
            },
        },
        "yolov9_c": {
            "csp_channels": [[1024, 256, 256], [768, 512, 512], [1024, 512, 512]],
            "adown_channels": [[256, 256], [512, 512]],
            "concat_sources": [[-1, "N4"], [-1, "N3"]],
            "aux_cfg": {
                "cblinear_channels": [[512, [256]], [512, [256, 512]], [512, [256, 512, 512]]],
                "csp_channels": [[128, 256, 128], [256, 512, 256], [512, 512, 512], [512, 512, 512]],
                "conv_channels": [[3, 64], [64, 128]],
                "adown_channels": [[], [256, 256], [512, 512], [512, 512]],
                "cbfuse_indices": [[], [0, 0, 0], [1, 1], [2]],
                "cbfuse_sources": [[], ["R3", "R4", "R5", -1], ["R4", "R5", -1], ["R5", -1]],
            },
        },
    }

    def __new__(cls, model_name: str, num_classes: int) -> YOLOHeadModule:
        """Constructor for YOLOHead for v7 and v9."""
        if model_name not in cls.YOLOHEAD_CFG:
            msg = f"model type '{model_name}' is not supported"
            raise KeyError(msg)

        return YOLOHeadModule(
            **cls.YOLOHEAD_CFG[model_name],
            num_classes=num_classes,
        )
