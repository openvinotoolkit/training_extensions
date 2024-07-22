# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTDetr model implementations."""

from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np
import torch
import torchvision
from torch import Tensor, nn

from otx.algo.detection.backbones import PResNet
from otx.algo.detection.heads import RTDETRTransformer
from otx.algo.detection.losses import DetrCriterion
from otx.algo.detection.necks import HybridEncoder
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.detection import ExplainableOTXDetModel


class DETR(nn.Module):
    """DETR model."""

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        criterion: nn.Module | None = None,
        optimizer_configuration: list[dict] | None = None,
        multi_scale: list[int] | None = None,
        num_top_queries: int = 300,
    ) -> None:
        """DETR model implementation.

        Args:
            backbone (nn.Module): Backbone module.
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            num_classes (int): Number of classes.
            criterion (nn.Module, optional): Loss function.
                If None then default DetrCriterion is used.
            optimizer_configuration (list[dict], optional): Optimizer configuration.
                Defaults to None.
            multi_scale (list[int], optional): List of image sizes.
                Defaults to None.
            num_top_queries (int, optional): Number of top queries to return.
                Defaults to 300.
        """
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = (
            multi_scale
            if multi_scale is not None
            else [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
        )
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.criterion = (
            criterion
            if criterion is not None
            else DetrCriterion(
                weight_dict={"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
                num_classes=num_classes,
                gamma=2.0,
                alpha=0.75,
            )
        )
        self.optimizer_configuration = optimizer_configuration

    def _forward_features(self, images: Tensor, targets: dict[str, Any] | None = None) -> dict[str, Tensor]:
        images = self.backbone(images)
        images = self.encoder(images)
        return self.decoder(images, targets)

    def forward(self, images: Tensor, targets: dict[str, Any] | None = None) -> dict[str, Tensor] | Tensor:
        """Forward pass of the model."""
        if self.multi_scale and self.training:
            sz = int(np.random.choice(self.multi_scale))
            images = nn.functional.interpolate(images, size=[sz, sz])

        output = self._forward_features(images, targets)
        if self.training:
            return self.criterion(output, targets)
        return output

    def export(
        self,
        batch_inputs: Tensor,
        batch_img_metas: list[dict],
        explain_mode: bool = False,
    ) -> dict[str, Any] | tuple[list[Any], list[Any], list[Any]]:
        """Exports the model."""
        if explain_mode:
            msg = "Explain mode is not supported for DETR models yet."
            raise NotImplementedError(msg)
        return self.postprocess(self._forward_features(batch_inputs), deploy_mode=True)

    def postprocess(
        self,
        outputs: dict[str, Tensor],
        original_size: tuple[int, int] | None = None,
        deploy_mode: bool = False,
    ) -> dict[str, Tensor] | tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Post-processes the model outputs.

        Args:
            outputs (dict[str, Tensor]): The model outputs.
            original_size (tuple[int, int], optional): The original size of the input images. Defaults to None.
            deploy_mode (bool, optional): Whether to run in deploy mode. Defaults to False.

        Returns:
            dict[str, Tensor] | tuple[list[Tensor], list[Tensor], list[Tensor]]: The post-processed outputs.
        """
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]

        # convert bbox to xyxy and rescale back to original size (resize in OTX)
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        if not deploy_mode and original_size is not None:
            original_size_tensor = torch.tensor(original_size).to(bbox_pred.device)
            bbox_pred *= original_size_tensor.repeat(1, 2).unsqueeze(1)

        # perform scores computation and gather topk results
        scores = nn.functional.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
        labels = index % self.num_classes
        index = index // self.num_classes
        boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        if deploy_mode:
            return {"bboxes": boxes, "labels": labels, "scores": scores}

        scores_list, boxes_list, labels_list = [], [], []

        for sc, bb, ll in zip(scores, boxes, labels):
            scores_list.append(sc)
            boxes_list.append(
                torchvision.tv_tensors.BoundingBoxes(bb, format="xyxy", canvas_size=original_size),
            )
            labels_list.append(ll.long())

        return scores_list, boxes_list, labels_list


class RTDETR(ExplainableOTXDetModel):
    """RTDETR model."""

    image_size = (1, 3, 640, 640)
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (255.0, 255.0, 255.0)
    load_from: str | None = None

    def _customize_inputs(
        self,
        entity: DetBatchDataEntity,
        pad_size_divisor: int = 32,
        pad_value: int = 0,
    ) -> dict[str, Any]:
        return {
            "images": entity.images,
            "targets": [{"boxes": bb, "labels": ll} for bb, ll in zip(entity.bboxes, entity.labels)],
        }

    def _customize_outputs(
        self,
        outputs: list[torch.Tensor] | dict,
        inputs: DetBatchDataEntity,
    ) -> DetBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if isinstance(v, list):
                    losses[k] = sum(v)
                elif isinstance(v, Tensor):
                    losses[k] = v
                else:
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        scores, bboxes, labels = self.model.postprocess(outputs, [img_info.img_shape for img_info in inputs.imgs_info])

        return DetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Configure an optimizer and learning-rate schedulers.

        Configure an optimizer and learning-rate schedulers
        from the given optimizer and scheduler or scheduler list callable in the constructor.
        Generally, there is two lr schedulers. One is for a linear warmup scheduler and
        the other is the main scheduler working after the warmup period.

        Returns:
            Two list. The former is a list that contains an optimizer
            The latter is a list of lr scheduler configs which has a dictionary format.
        """
        param_groups = self._get_optim_params(self.model.optimizer_configuration, self.model)
        optimizer = self.optimizer_callable(param_groups)
        schedulers = self.scheduler_callable(optimizer)

        def ensure_list(item: Any) -> list:  # noqa: ANN401
            return item if isinstance(item, list) else [item]

        lr_scheduler_configs = []
        for scheduler in ensure_list(schedulers):
            lr_scheduler_config = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                lr_scheduler_config["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                lr_scheduler_config["monitor"] = scheduler.monitor
            lr_scheduler_configs.append(lr_scheduler_config)

        return [optimizer], lr_scheduler_configs

    @staticmethod
    def _get_optim_params(cfg: list[dict[str, Any]] | None, model: nn.Module) -> list[dict[str, Any]]:
        """Perform no bias decay and learning rate correction for the modules.

        The configuration dict should consist of regular expression pattern for the model parameters with "params" key.
        Other optimizer parameters can be added as well.

        E.g.:
            cfg = [{"params": "^((?!b).)*$", "lr": 0.01, "weight_decay": 0.0}, ..]
            The above configuration is for the parameters that do not contain "b".

            ^(?=.*a)(?=.*b).*$         means including a and b
            ^((?!b.)*a((?!b).)*$       means including a but not b
            ^((?!b|c).)*a((?!b|c).)*$  means including a but not (b | c)
        """
        if cfg is None:
            return model.parameters()

        cfg = copy.deepcopy(cfg)

        param_groups = []
        visited = []
        for pg in cfg:
            if "params" not in pg:
                msg = f"The 'params' key should be included in the configuration, but got {pg.keys()}"
                raise ValueError(msg)
            pattern = pg["params"]
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg["params"] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({"params": params.values()})
            visited.extend(list(params.keys()))

        return param_groups

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.image_size is None:
            raise ValueError(self.image_size)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=self.mean,
            std=self.std,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "dynamic_axes": {
                    "images": {0: "batch"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "scores": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
                "opset_version": 16,
            },
            output_names=["bboxes", "labels", "scores"],
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for RT-DETR."""
        return {"model_type": "transformer"}


class RTDETR18(RTDETR):
    """RT-DETR with ResNet-18 backbone."""

    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth"
    )

    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(
            depth=18,
            pretrained=True,
            return_idx=[1, 2, 3],
        )
        encoder = HybridEncoder(
            in_channels=[128, 256, 512],
            expansion=0.5,
            eval_spatial_size=self.image_size[2:],
        )
        decoder = RTDETRTransformer(
            num_classes=num_classes,
            num_decoder_layers=3,
            feat_channels=[256, 256, 256],
            eval_spatial_size=self.image_size[2:],
        )

        optimizer_configuration = [
            # no weight decay for norm layers in backbone
            {"params": "^(?=.*backbone)(?=.*norm).*$", "weight_decay": 0.0, "lr": 0.00001},
            # lr for the backbone, but not norm layers is 0.00001
            {"params": "^(?=.*backbone)(?!.*norm).*$", "lr": 0.00001},
            # no weight decay for norm layers and biases in encoder and decoder layers
            {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$", "weight_decay": 0.0},
        ]

        return DETR(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            num_classes=num_classes,
            optimizer_configuration=optimizer_configuration,
        )


class RTDETR50(RTDETR):
    """RT-DETR with ResNet-50 backbone."""

    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth"
    )

    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(
            depth=50,
            return_idx=[1, 2, 3],
            pretrained=True,
            freeze_at=0,
            norm_cfg={"type": "FBN", "name": "norm"},
        )
        encoder = HybridEncoder(
            eval_spatial_size=self.image_size[2:],
        )
        decoder = RTDETRTransformer(
            num_classes=num_classes,
            feat_channels=[256, 256, 256],
            eval_spatial_size=self.image_size[2:],
            num_decoder_layers=6,
        )

        optimizer_configuration = [
            # lr for all layers in backbone is 0.00001
            {"params": "backbone", "lr": 0.00001},
            # no weight decay for norm layers and biases in decoder
            {"params": "^(?=.*decoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
            # no weight decay for norm layers and biases in encoder
            {"params": "^(?=.*encoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
        ]

        return DETR(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            num_classes=num_classes,
            optimizer_configuration=optimizer_configuration,
        )


class RTDETR101(RTDETR):
    """RT-DETR with ResNet-101 backbone."""

    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth"
    )

    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(
            depth=101,
            return_idx=[1, 2, 3],
            norm_cfg={"type": "FBN", "name": "norm"},
            pretrained=True,
            freeze_at=0,
        )

        encoder = HybridEncoder(
            hidden_dim=384,
            dim_feedforward=2048,
            in_channels=[512, 1024, 2048],
            eval_spatial_size=self.image_size[2:],
        )

        decoder = RTDETRTransformer(
            num_classes=num_classes,
            feat_channels=[384, 384, 384],
            eval_spatial_size=self.image_size[2:],
        )

        # no bias decay and learning rate correction for the backbone.
        # Without this correction gradients explosion will take place.
        optimizer_configuration = [
            # lr for all layers in backbone is 0.000001
            {"params": "backbone", "lr": 0.000001},
            # no weight decay for norm layers and biases in encoder
            {"params": "^(?=.*encoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
            # no weight decay for norm layers and biases in decoder
            {"params": "^(?=.*decoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
        ]

        return DETR(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            num_classes=num_classes,
            optimizer_configuration=optimizer_configuration,
        )
