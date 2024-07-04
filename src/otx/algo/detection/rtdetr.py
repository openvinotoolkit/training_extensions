"""by lyuwenyu
"""

from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.base import OTXBatchLossEntity
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
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
        self.bbox_head = None
        self.multi_scale = multi_scale

    def forward(self, images, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            images = F.interpolate(images, size=[sz, sz])

        images = self.backbone(images)
        images = self.encoder(images)
        output_losses = self.decoder(images, targets)

        return output_losses

    def init_weights(self):
        self.backbone.init_weights()
        self.encoder.init_weights()
        self.decoder.init_weights()


class OTX_RTDETR(ExplainableOTXDetModel):
    def _customize_inputs(
        self,
        entity: DetBatchDataEntity,
        pad_size_divisor: int = 32,
        pad_value: int = 114,  # YOLOX uses 114 as pad_value
    ) -> dict[str, Any]:

        inputs = super()._customize_inputs(entity=entity, pad_size_divisor=pad_size_divisor, pad_value=pad_value)
        entity = inputs["entity"]
        return {"images": entity.images, "targets": {"boxes": entity.bboxes, "labels": entity.labels}}

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

        scores = []
        bboxes = []
        labels = []
        saliency_map = []
        feature_vector = []

        for out in outputs:
            scores.append(out["scores"])
            bboxes.append(out["boxes"])
            labels.append(out["labels"])
            if "saliency_map" in out:
                saliency_map.append(out["saliency_map"].detach().cpu().numpy())
            if "feature_vector" in out:
                feature_vector.append(out["feature_vector"].detach().cpu().numpy())

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

    @staticmethod
    def load_from_checkpoint(model, checkpoint_path: str, **kwargs):
        state_dict = torch.load(checkpoint_path)["model"]
        model.load_state_dict(state_dict, strict=False)


class OTX_RTDETR_18(OTX_RTDETR):
    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(depth=18, pretrained=True, freeze_at=-1, return_idx=[1, 2, 3], num_stages=4, freeze_norm=False)
        encoder = HybridEncoder(in_channels=[128, 256, 512], feat_strides=[8, 16, 32], hidden_dim=256, expansion=0.5)
        decoder = RTDETRTransformer(num_classes=num_classes, eval_idx=-1, num_decoder_layers=3, num_denoising=100,
                                      feat_channels=[256, 256, 256], feat_strides=[8, 16, 32], hidden_dim=256, num_levels=3,
                                      num_queries=300, eval_spatial_size=[640, 640])
        self.load_from = "/home/kprokofi/RT-DETR-2/rtdetr_pytorch/output/rtdetr_r18vd_6x_coco/checkpoint0070.pth"
        return RTDETR(backbone, encoder, decoder)
