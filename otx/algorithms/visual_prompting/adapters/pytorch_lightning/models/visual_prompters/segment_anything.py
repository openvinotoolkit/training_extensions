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
from typing import Any, Dict, List, Tuple
from torch import optim

import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.decoders import (
    SAMMaskDecoder,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.encoders import (
    SAMImageEncoderViT,
    SAMPromptEncoder,
)

from pytorch_lightning import LightningModule


CKPT_PATHS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
}


class SegmentAnything(LightningModule):
    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
        checkpoint: str = None,
        mask_threshold: float = 0.,
        return_logits: bool = False
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
            image_encoder (nn.Module): The backbone used to encode the image into image embeddings that allow for efficient mask prediction.
            prompt_encoder (nn.Module): Encodes various types of input prompts.
            mask_decoder (nn.Module): Predicts masks from the image embeddings and encoded prompts.
            freeze_image_encoder (bool): Whether freezing image encoder, default is True.
            freeze_prompt_encoder (bool): Whether freezing prompt encoder, default is True.
            freeze_mask_decoder (bool): Whether freezing mask decoder, default is False.
            checkpoint (optional, str): Checkpoint path to be loaded, default is None.
            mask_threshold (float): 
        """
        super().__init__()
        # self.save_hyperparameters()

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.mask_threshold = mask_threshold
        self.return_logits = return_logits

        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        if freeze_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

        if freeze_mask_decoder:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False

        self.train_iou = torchmetrics.classification.BinaryJaccardIndex()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()
        self.val_iou = torchmetrics.classification.BinaryJaccardIndex()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()
        if checkpoint:
            try:
                self.load_from_checkpoint(checkpoint)
            except:
                if str(checkpoint).startswith("http"):
                    state_dict = torch.hub.load_state_dict_from_url(str(checkpoint))
                else:
                    with open(checkpoint, "rb") as f:
                        state_dict = torch.load(f)
                self.load_state_dict(state_dict)

    def forward(self, images, bboxes, points=None):
        _, _, height, width = images.shape
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

            masks = F.interpolate(
                low_res_masks,
                (height, width),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        return pred_masks, ious

    def training_step(self, batch, batch_idx):
        """Training step of SAM."""
        images = batch["images"]
        bboxes = batch["bboxes"]
        points = batch["points"]
        gt_masks = batch["masks"]

        pred_masks, ious = self(images, bboxes, points)

        loss_focal = 0.
        loss_dice = 0.
        loss_iou = 0.
        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)

        for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, ious):
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
        gt_masks = batch["masks"]

        pred_masks, _ = self(images, bboxes, points)
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
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

        masks = self.postprocess_masks(pred_masks, self.input_size, self.original_size)
        if not self.return_logits:
            masks = masks > self.mask_threshold

        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
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


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = SegmentAnything(
        image_encoder=SAMImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=SAMPromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=SAMMaskDecoder(
            num_multimask_outputs=3,
            transformer_cfg=dict(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        checkpoint=checkpoint
    )

    return sam


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}
