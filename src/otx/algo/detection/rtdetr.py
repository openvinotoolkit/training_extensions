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

class RTDETRPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']

    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self

    @property
    def iou_types(self, ):
        return ('bbox', )


class RTDETR(nn.Module):
    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module, num_classes:int, criterion: nn.Module,
                 postprocessor: nn.Module, multi_scale: list[int] | None = None, num_top_queries=300):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.criterion = criterion
        self.postprocessor = postprocessor

    def forward(self, images, targets=None):
        original_size = images.shape[-2:]

        if self.multi_scale and self.training:
            sz = int(np.random.choice(self.multi_scale))
            images = F.interpolate(images, size=[sz, sz])

        images = self.backbone(images)
        images = self.encoder(images)
        output = self.decoder(images, targets)

        if self.training:
            return self.criterion(output, targets)

        return self.postprocess(output, original_size)


    def postprocess(self, outputs, original_size):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        # convert bbox to xyxy and rescale back to original size (resize in OTX)
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        original_size = torch.tensor(original_size).to(bbox_pred.device)
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
        backbone = PResNet(depth=18, pretrained=True, freeze_at=-1, return_idx=[1, 2, 3], num_stages=4, freeze_norm=False)
        encoder = HybridEncoder(in_channels=[128, 256, 512], feat_strides=[8, 16, 32], hidden_dim=256, expansion=0.5, dim_feedforward=1024, eval_spatial_size=[640, 640])
        decoder = RTDETRTransformer(num_classes=num_classes, eval_idx=-1, num_decoder_layers=3, num_denoising=100,
                                      feat_channels=[256, 256, 256], feat_strides=[8, 16, 32], hidden_dim=256, num_levels=3,
                                      num_queries=300, eval_spatial_size=[640, 640], aux_loss=True)
        postprocessor = RTDETRPostProcessor(num_classes=num_classes, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False)
        criterion = RTDetrCriterion(weight_dict={'loss_vfl': 1.0, 'loss_bbox': 5, 'loss_giou': 2},
                                    losses=['vfl', 'boxes'], num_classes=num_classes, gamma=2.0, alpha=0.75)

        return RTDETR(backbone=backbone,
                      encoder=encoder,
                      decoder=decoder,
                      num_classes=num_classes,
                      postprocessor=postprocessor,
                      criterion=criterion,
                      multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800])


class OTX_RTDETR_50(OTX_RTDETR):
    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth"
    )
    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(depth=50, return_idx=[1, 2, 3], num_stages=4, freeze_norm=True, pretrained=True, freeze_at=0)
        encoder = HybridEncoder(in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], hidden_dim=256, expansion=1.0, dim_feedforward=1024, eval_spatial_size=[640, 640])
        decoder = RTDETRTransformer(num_classes=num_classes, feat_channels=[256, 256, 256], feat_strides=[8, 16, 32], hidden_dim=256,
                                    num_levels=3, num_queries=300, eval_spatial_size=[640, 640], num_decoder_layers=6, num_denoising=100, eval_idx=-1)
        criterion = RTDetrCriterion(weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
                                    losses=['vfl', 'boxes'], num_classes=num_classes, gamma=2.0, alpha=0.75)
        postprocessor = RTDETRPostProcessor(num_classes=num_classes, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False)

        return RTDETR(backbone=backbone,
                      encoder=encoder,
                      decoder=decoder,
                      num_classes=num_classes,
                      postprocessor=postprocessor,
                      criterion=criterion,
                      multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800])


class OTX_RTDETR_101(OTX_RTDETR):
    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth"
    )
    def _build_model(self, num_classes: int) -> nn.Module:
        num_classes = 1

        backbone = PResNet(depth=101, return_idx=[1, 2, 3], num_stages=4, freeze_norm=True, pretrained=True, freeze_at=0)

        encoder = HybridEncoder(hidden_dim=384, dim_feedforward=2048, in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32],
                                expansion=1.0, eval_spatial_size=[640, 640])

        decoder = RTDETRTransformer(num_classes=num_classes, feat_channels=[384, 384, 384], feat_strides=[8, 16, 32], hidden_dim=256,
                                    num_levels=3, num_queries=300, eval_spatial_size=[640, 640], num_decoder_layers=6, nhead=8, num_denoising=100, eval_idx=-1)

        criterion = RTDetrCriterion(weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
                                    losses=['vfl', 'boxes'], num_classes=num_classes, gamma=2.0, alpha=0.75)

        postprocessor = RTDETRPostProcessor(num_classes=num_classes, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False)

        return RTDETR(backbone=backbone,
                      encoder=encoder,
                      decoder=decoder,
                      num_classes=num_classes,
                      criterion=criterion,
                      postprocessor=postprocessor,
                      multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800])
