"""by lyuwenyu
"""

from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.base import OTXBatchLossEntity
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any

from otx.core.model.detection import ExplainableOTXDetModel
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.algo.detection.backbones import PResNet
from otx.algo.detection.necks import HybridEncoder
from otx.algo.detection.heads import RTDETRTransformer
from otx.algo.detection.losses import RTDetrCriterion

import numpy as np


__all__ = ['RTDETR', ]


class RTDETR(nn.Module):
    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.loss = RTDetrCriterion({"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2}, ['vfl', 'boxes'])

    def forward(self, images, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            images = F.interpolate(images, size=[sz, sz])

        images = self.backbone(images)
        breakpoint()
        images = self.encoder(images)
        breakpoint()
        outputs = self.decoder(images, targets)

        breakpoint()
        return outputs


class OTX_RTDETR(ExplainableOTXDetModel):
    def _customize_inputs(
        self,
        entity: DetBatchDataEntity,
        pad_size_divisor: int = 32,
        pad_value: int = 114,  # YOLOX uses 114 as pad_value
    ) -> dict[str, Any]:

        entity = super()._customize_inputs(entity=entity, pad_size_divisor=pad_size_divisor, pad_value=pad_value)
        return {"images": entity.images, "targets": {"boxes": entity.bboxes, "labels": entity.labels}}

    def _customize_outputs(self, outputs: list[InstanceData] | dict, inputs: DetBatchDataEntity) -> DetBatchPredEntity | OTXBatchLossEntity:

        scores = outputs["scores"]
        bboxes = outputs["bboxes"]
        labels = outputs["labels"]
        saliency_map = outputs["saliency_map"].detach().cpu().numpy()
        feature_vector = outputs["feature_vector"].detach().cpu().numpy()

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


class OTX_RTDETR_18(OTX_RTDETR):
    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(depth=18)
        encoder = HybridEncoder()
        decoder = RTDETRTransformer(num_classes)

        return RTDETR(backbone, encoder, decoder)
