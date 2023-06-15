"""SAM module for visual prompting.

paper: https://arxiv.org/abs/2304.02643
reference: https://github.com/facebookresearch/segment-anything
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import re
from collections import OrderedDict
import torchmetrics
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.decoders import (
    SAMMaskDecoder,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.encoders import (
    SAMImageEncoder,
    SAMPromptEncoder,
)

CKPT_PATHS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
}


class SegmentAnything(LightningModule):
    def __init__(self, config: DictConfig, config_optimizer: DictConfig) -> None:
        """SAM predicts object masks from an image and input prompts."""
        super().__init__()
        self.config_optimizer = config_optimizer
        # self.save_hyperparameters()

        # TODO (sungchul): Currently, backbone is assumed as vit. Depending on backbone, image_embedding_size can be changed.
        if "vit" in config.backbone:
            patch_size = 16
            self.image_embedding_size = config.image_size // patch_size
        else:
            raise NotImplementedError((
                f"{config.backbone} for image encoder of SAM is not implemented yet. "
                f"Use vit_b, l, or h."
            ))

        self.image_encoder = SAMImageEncoder(config)
        self.prompt_encoder = SAMPromptEncoder(
            embed_dim=256,
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=(config.image_size, config.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = SAMMaskDecoder(
            num_multimask_outputs=3,
            transformer_cfg=dict(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.mask_threshold = config.mask_threshold
        self.return_logits = config.return_logits

        if config.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        if config.freeze_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

        if config.freeze_mask_decoder:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False

        self.train_iou = torchmetrics.classification.BinaryJaccardIndex()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()
        self.val_iou = torchmetrics.classification.BinaryJaccardIndex()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()
        if config.checkpoint:
            try:
                self.load_from_checkpoint(config.checkpoint)
            except:
                if str(config.checkpoint).startswith("http"):
                    state_dict = torch.hub.load_state_dict_from_url(str(config.checkpoint))
                    for p, r in [(r'^image_encoder\.', r'image_encoder.backbone.')]:
                        state_dict = OrderedDict({
                            re.sub(p, r, k): v for k, v in state_dict.items()})
                else:
                    with open(config.checkpoint, "rb") as f:
                        state_dict = torch.load(f)
                self.load_state_dict(state_dict)

    def forward(self, images, bboxes, points=None):
        image_embeddings = self.image_encoder(images)
        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, bboxes):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=bbox,
                masks=None,
            )

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred_masks.append(low_res_masks)
            ious.append(iou_predictions)

        return pred_masks, ious

    def training_step(self, batch, batch_idx):
        """Training step of SAM."""
        images = batch["images"]
        bboxes = batch["bboxes"]
        points = batch["points"]
        gt_masks = batch["gt_masks"]

        pred_masks, ious = self(images, bboxes, points)

        loss_focal = 0.
        loss_dice = 0.
        loss_iou = 0.
        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
        for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, ious):
            pred_mask = self.postprocess_masks(pred_mask, images.shape[2:])
            self.train_iou(pred_mask, gt_mask)
            self.train_f1(pred_mask, gt_mask)
            pred_mask = pred_mask.flatten(1)
            gt_mask = gt_mask.flatten(1).float()

            loss_focal += self.calculate_sigmoid_focal_loss(pred_mask, gt_mask, num_masks)
            loss_dice += self.calculate_dice_loss(pred_mask, gt_mask, num_masks)
            batch_iou = self.calculate_iou(pred_mask, gt_mask)
            loss_iou += F.mse_loss(iou_prediction, batch_iou.unsqueeze(1), reduction='sum') / num_masks

        loss = 20. * loss_focal + loss_dice + loss_iou
        results = dict(
            train_IoU=self.train_iou,
            train_F1=self.train_f1,
            train_loss=loss,
            train_loss_focal=loss_focal,
            train_loss_dice=loss_dice,
            train_loss_iou=loss_iou)
        self.log_dict(results, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step of SAM."""
        images = batch["images"]
        bboxes = batch["bboxes"]
        points = batch["points"]
        gt_masks = batch["gt_masks"]

        pred_masks, _ = self(images, bboxes, points)
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            pred_mask = self.postprocess_masks(pred_mask, images.shape[2:])
            self.val_iou(pred_mask, gt_mask)
            self.val_f1(pred_mask, gt_mask)

        results = dict(val_IoU=self.val_iou, val_F1=self.val_f1)
        self.log_dict(results, on_epoch=True, prog_bar=True)
        return results

    def predict_step(self, batch, batch_idx):
        """Predict step of SAM."""
        images = batch["images"]
        bboxes = batch["bboxes"]
        points = batch["points"]

        pred_masks, _ = self(images, bboxes, points)

        masks = [
            self.postprocess_masks(pred_mask, images[i].shape[2:], batch["original_size"][i], is_predict=True) for i, pred_mask in enumerate(pred_masks)
        ]
        if not self.return_logits:
            masks = masks > self.mask_threshold

        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Optional[Tuple[int, ...]] = None,
        is_predict: bool = False
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (torch.Tensor): Batched masks from the mask_decoder, in BxCxHxW format.
            input_size (tuple(int, int)): The size of the image input to the model, in (H, W) format. Used to remove padding.
            original_size (tuple(int, int)): The original size of the image before resizing for input to the model, in (H, W) format.
            is_predict (bool, optional): ...

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(masks, input_size, mode="bilinear", align_corners=False)
        if is_predict:
            masks = masks[..., : input_size[0], : input_size[1]]
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks.squeeze(1)
    
    def configure_optimizers(self):
        name = self.config_optimizer.pop("name")
        optimizer = getattr(optim, name)(self.parameters(), **self.config_optimizer)
        return optimizer
    
    def calculate_dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor, num_masks: torch.Tensor):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        
        Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/modeling_maskformer.py#L269

        Args:
            inputs (`torch.Tensor`):
                A tensor representing a mask.
            targets (`torch.Tensor`):
                A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
                (0 for the negative class and 1 for the positive class).
            num_masks (`int`):
                The number of masks present in the current batch, used for normalization.

        Returns:
            Loss tensor
        """
        inputs = inputs.sigmoid()
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks
    
    def calculate_sigmoid_focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2
    ):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Referece: https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/modeling_maskformer.py#L300

        Args:
            inputs (`torch.Tensor`):
                A float tensor of arbitrary shape.
            targets (`torch.Tensor`):
                A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
                (0 for the negative class and 1 for the positive class).
            num_masks (`int`):
                The number of masks present in the current batch, used for normalization.
            alpha (float, *optional*, defaults to 0.25):
                Weighting factor in range (0,1) to balance positive vs negative examples.
            gamma (float, *optional*, defaults to 2.0):
                Exponent of the modulating factor \\(1 - p_t\\) to balance easy vs hard examples.

        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_masks
    
    def calculate_iou(self, inputs: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-7):
        """"""
        pred_mask = (inputs >= 0.5).float()
        intersection = torch.sum(torch.mul(pred_mask, targets), dim=1)
        union = torch.sum(pred_mask, dim=1) + torch.sum(targets, dim=1) - intersection
        iou = intersection / (union + epsilon)
        return iou
