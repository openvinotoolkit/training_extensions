"""by lyuwenyu
"""

from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.base import OTXBatchLossEntity
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from typing import Any
import torchvision

from otx.core.model.detection import ExplainableOTXDetModel
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.algo.detection.backbones import PResNet
from otx.algo.detection.necks import HybridEncoder
from otx.algo.detection.heads import RTDETRTransformer
from otx.algo.detection.losses import RTDetrCriterion

import numpy as np


__all__ = ['RTDETR', ]


class RTDETR(nn.Module):
    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module, num_classes:int, multi_scale: list[int] | None = None, num_top_queries=300):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries

    def forward(self, images, targets=None):
        original_size = images.shape[-2:]
        if self.multi_scale and self.training:
            sz = int(np.random.choice(self.multi_scale))
            images = F.interpolate(images, size=[sz, sz])

        images = self.backbone(images)
        images = self.encoder(images)
        output = self.decoder(images, targets)

        if self.training:
            return output
        return self.postprocess(output, torch.tensor(original_size).to(images[-1].device))

    def postprocess(self, outputs, original_size):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        # convert bbox to xyxy and rescale back to original size (resize in OTX)
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= original_size.repeat(1, 2).unsqueeze(1)

        # perform scores computation and gather topk results
        scores = F.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
        labels = index % self.num_classes
        index = index // self.num_classes
        boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        scores_list = []
        boxes_list = []
        labels_list = []

        for sc, bb, ll in zip(scores, boxes, labels):
            scores_list.append(sc)
            boxes_list.append(bb)
            labels_list.append(ll)

        return scores_list, boxes_list, labels_list

    def init_weights(self):
        self.backbone.init_weights()
        self.encoder.init_weights()
        self.decoder.init_weights()


class OTX_RTDETR(ExplainableOTXDetModel):
    def _customize_inputs(
        self,
        entity: DetBatchDataEntity,
        pad_size_divisor: int = 32,
        pad_value: int = 0,
    ) -> dict[str, Any]:

        return {"images": entity.images, "targets": [{"boxes" : bb, "labels": ll} for bb, ll in zip(entity.bboxes, entity.labels)]}

    def _customize_outputs(self, outputs: list[InstanceData] | dict, inputs: DetBatchDataEntity) -> DetBatchPredEntity | OTXBatchLossEntity:

        if self.training:
            targets = [{"boxes" : bb, "labels": ll} for bb, ll in zip(inputs.bboxes, inputs.labels)]
            outputs_losses = self.criterion(outputs, targets)
            if not isinstance(outputs_losses, dict):
                raise TypeError(outputs_losses)

            losses = OTXBatchLossEntity()
            for k, v in outputs_losses.items():
                if isinstance(v, list):
                    losses[k] = sum(v)
                elif isinstance(v, Tensor):
                    losses[k] = v
                else:
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        saliency_map = [] # TODO add saliency map and XAI feature
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


class OTX_RTDETR_18(OTX_RTDETR):
    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth"
    )
    def _build_model(self, num_classes: int) -> nn.Module:
        num_classes = 20
        # backbone = PResNet(depth=18, freeze_at=-1, freeze_norm=False)
        # encoder = HybridEncoder(in_channels=[128, 256, 512], hidden_dim=256, expansion=0.5)
        # decoder = RTDETRTransformer(num_classes=num_classes, num_decoder_layers=3, num_denoising=100)
        self.criterion = RTDetrCriterion(weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
                                    losses=['vfl', 'boxes'], num_classes=num_classes)
        backbone = PResNet(depth=18, pretrained=True, freeze_at=-1, return_idx=[1, 2, 3], num_stages=4, freeze_norm=False)
        encoder = HybridEncoder(in_channels=[128, 256, 512], feat_strides=[8, 16, 32], hidden_dim=256, expansion=0.5)
        decoder = RTDETRTransformer(num_classes=num_classes, eval_idx=-1, num_decoder_layers=3, num_denoising=100,
                                      feat_channels=[256, 256, 256], feat_strides=[8, 16, 32], hidden_dim=256, num_levels=3,
                                      num_queries=300, eval_spatial_size=[640, 640])

        return RTDETR(backbone=backbone,
                      encoder=encoder,
                      decoder=decoder,
                      num_classes=num_classes,
                      multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800])


class OTX_RTDETR_50(OTX_RTDETR):
    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth"
    )
    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(depth=50)
        encoder = HybridEncoder()
        decoder = RTDETRTransformer(num_classes=num_classes)
        self.criterion = RTDetrCriterion(weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
                                    losses=['vfl', 'boxes'], num_classes=num_classes)

        return RTDETR(backbone=backbone,
                      encoder=encoder,
                      decoder=decoder,
                      num_classes=num_classes,
                      multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800])


class OTX_RTDETR_101(OTX_RTDETR):
    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth"
    )
    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(depth=101)
        encoder = HybridEncoder(hidden_dim=384, dim_feedforward=2048)
        decoder = RTDETRTransformer(num_classes=num_classes, feat_channels=[384, 384, 384])
        criterion = RTDetrCriterion(weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
                                    losses=['vfl', 'boxes'], num_classes=num_classes)

        return RTDETR(backbone=backbone,
                      encoder=encoder,
                      decoder=decoder,
                      criterion=criterion,
                      num_classes=num_classes,
                      multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800])


        self.load_from = "/home/kprokofi/RT-DETR-2/rtdetr_pytorch/output/rtdetr_r18vd_6x_coco/checkpoint0070.pth"
        return RTDETR(backbone, encoder, decoder)
