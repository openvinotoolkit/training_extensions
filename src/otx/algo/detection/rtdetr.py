"""by lyuwenyu
"""

import copy
import re
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

from otx.algo.detection.backbones import PResNet
from otx.algo.detection.heads import RTDETRTransformer
from otx.algo.detection.losses import RTDetrCriterion
from otx.algo.detection.necks import HybridEncoder
from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.detection import OTXDetectionModel, ExplainableOTXDetModel

__all__ = ["RTDETR"]


class RTDETR(nn.Module):
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
    ):
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
        default_criterion = RTDetrCriterion(
            weight_dict={"loss_vfl": 1.0, "loss_bbox": 5, "loss_giou": 2},
            num_classes=num_classes,
            gamma=2.0,
            alpha=0.75,
        )
        self.criterion = criterion if criterion is not None else default_criterion
        self.optimizer_configuration = optimizer_configuration

    def _forward_features(self, images: Tensor, targets: dict[str, Any] | None = None):
        images = self.backbone(images)
        images = self.encoder(images)
        return self.decoder(images, targets)

    def forward(self, images: Tensor, targets: dict[str, Any] | None = None):
        original_size = images.shape[-2:]

        if self.multi_scale and self.training:
            sz = int(np.random.choice(self.multi_scale))
            images = F.interpolate(images, size=[sz, sz])

        output = self._forward_features(images, targets)
        if self.training:
            return self.criterion(output, targets)
        return self.postprocess(output, original_size)

    def postprocess(self, outputs, original_size=None, deploy_mode=False):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]

        # convert bbox to xyxy and rescale back to original size (resize in OTX)
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        if not deploy_mode and original_size is not None:
            original_size = torch.tensor(original_size).to(bbox_pred.device)
            bbox_pred *= original_size.repeat(1, 2).unsqueeze(1)

        # perform scores computation and gather topk results
        scores = F.sigmoid(logits)
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
                torchvision.tv_tensors.BoundingBoxes(bb, format="xyxy", canvas_size=original_size.tolist())
            )
            labels_list.append(ll.long())

        return scores_list, boxes_list, labels_list

    def export(
        self,
        batch_inputs: Tensor,
        batch_img_metas: list[dict],
    ) -> dict[str, Any]:
        return self.postprocess(self._forward_features(batch_inputs), deploy_mode=True)


class OTX_RTDETR(ExplainableOTXDetModel):
    image_size = (1, 3, 640, 640)
    mean = (0.0, 0.0, 0.0)
    std = (255.0, 255.0, 255.0)

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
        self, outputs: list[InstanceData] | dict, inputs: DetBatchDataEntity
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

        saliency_map = []  # TODO add saliency map and XAI feature
        feature_vector = []
        scores, bboxes, labels = outputs

        return DetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
            saliency_map=saliency_map,
            feature_vector=feature_vector,
        )

    def get_num_anchors(self) -> list[int]:
        """Gets the anchor configuration from model."""
        # TODO update anchor configuration

        return [1] * 10

    def configure_optimizers(self):
        """Configure an optimizer and learning-rate schedulers.

        Configure an optimizer and learning-rate schedulers
        from the given optimizer and scheduler or scheduler list callable in the constructor.
        Generally, there is two lr schedulers. One is for a linear warmup scheduler and
        the other is the main scheduler working after the warmup period.

        Returns:
            Two list. The former is a list that contains an optimizer
            The latter is a list of lr scheduler configs which has a dictionary format.
        """
        param_groups = self.get_optim_params(self.model.optimizer_configuration, self.model)
        optimizer = self.optimizer_callable(param_groups)
        optimizer = optimizer.__class__(optimizer.param_groups, **optimizer.defaults)
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
    def get_optim_params(cfg: list[dict[str, Any]] | None, model: nn.Module):
        """Perform no bias decay and learning rate correction for the modules.
        E.g.:
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
                    "images": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "scores": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
                "opset_version": 16,
            },
            output_names=["bboxes", "labels", "scores"]
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DinoV2Seg."""
        return {"model_type": "transformer"}


class OTX_RTDETR_18(OTX_RTDETR):
    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth"
    )

    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(
            depth=18, pretrained=True, freeze_at=-1, return_idx=[1, 2, 3], num_stages=4, freeze_norm=False
        )
        encoder = HybridEncoder(
            in_channels=[128, 256, 512],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            expansion=0.5,
            dim_feedforward=1024,
            eval_spatial_size=self.image_size[2:],
        )
        decoder = RTDETRTransformer(
            num_classes=num_classes,
            num_decoder_layers=3,
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            eval_spatial_size=self.image_size[2:],
        )
        criterion = RTDetrCriterion(
            weight_dict={"loss_vfl": 1.0, "loss_bbox": 5, "loss_giou": 2},
            losses=["vfl", "boxes"],
            num_classes=num_classes,
            gamma=2.0,
            alpha=0.75,
        )
        optimizer_configuration = [
            {"params": "^(?=.*backbone)(?=.*norm).*$", "weight_decay": 0.0, "lr": 0.00001},
            {"params": "^(?=.*backbone)(?!.*norm).*$", "lr": 0.00001},
            {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$", "weight_decay": 0.0},
        ]

        return RTDETR(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            num_classes=num_classes,
            criterion=criterion,
            optimizer_configuration=optimizer_configuration,
        )


class OTX_RTDETR_50(OTX_RTDETR):
    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth"
    )

    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(depth=50, return_idx=[1, 2, 3], num_stages=4, freeze_norm=True, pretrained=True, freeze_at=0)
        encoder = HybridEncoder(
            in_channels=[512, 1024, 2048],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            expansion=1.0,
            dim_feedforward=1024,
            eval_spatial_size=self.image_size[2:],
        )
        decoder = RTDETRTransformer(
            num_classes=num_classes,
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            num_levels=3,
            num_queries=300,
            eval_spatial_size=self.image_size[2:],
            num_decoder_layers=6,
            num_denoising=100,
            eval_idx=-1,
        )

        optimizer_configuration = [
            {"params": "backbone", "lr": 0.00001},
            {"params": "^(?=.*decoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
            {"params": "^(?=.*encoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
        ]

        return RTDETR(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            num_classes=num_classes,
            optimizer_configuration=optimizer_configuration,
        )


class OTX_RTDETR_101(OTX_RTDETR):
    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth"
    )

    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(
            depth=101, return_idx=[1, 2, 3], num_stages=4, freeze_norm=True, pretrained=True, freeze_at=0
        )

        encoder = HybridEncoder(
            hidden_dim=384,
            dim_feedforward=2048,
            in_channels=[512, 1024, 2048],
            feat_strides=[8, 16, 32],
            expansion=1.0,
            eval_spatial_size=self.image_size[2:],
        )

        decoder = RTDETRTransformer(
            num_classes=num_classes,
            feat_channels=[384, 384, 384],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            num_levels=3,
            num_queries=300,
            eval_spatial_size=self.image_size[2:],
            num_decoder_layers=6,
            nhead=8,
            num_denoising=100,
            eval_idx=-1,
        )

        # no bias decay and learning rate correction for the backbone. Without this correction gradients explosion will take place.
        optimizer_configuration = [
            {"params": "backbone", "lr": 0.000001},
            {"params": "^(?=.*encoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
            {"params": "^(?=.*decoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
        ]

        return RTDETR(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            num_classes=num_classes,
            optimizer_configuration=optimizer_configuration,
        )
