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

import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor, optim
from torch.nn import functional as F
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, Dice

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
    def __init__(self, config: DictConfig, state_dict: Optional[OrderedDict] = None) -> None:
        """SAM predicts object masks from an image and input prompts."""
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.set_models()
        self.freeze_networks()
        self.set_metrics()
        self.load_checkpoint(state_dict=state_dict)

    def set_models(self) -> None:
        """"""
        # TODO (sungchul): Currently, backbone is assumed as vit. Depending on backbone, image_embedding_size can be changed.
        if "vit" in self.config.model.backbone:
            patch_size = 16
            self.image_embedding_size = self.config.model.image_size // patch_size
        else:
            raise NotImplementedError((
                f"{self.config.model.backbone} for image encoder of SAM is not implemented yet. "
                f"Use vit_b, l, or h."
            ))

        self.image_encoder = SAMImageEncoder(self.config.model)
        self.prompt_encoder = SAMPromptEncoder(
            embed_dim=256,
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=(self.config.model.image_size, self.config.model.image_size),
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

    def freeze_networks(self) -> None:
        """"""
        if self.config.model.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        if self.config.model.freeze_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

        if self.config.model.freeze_mask_decoder:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False

    def set_metrics(self) -> None:
        """"""
        assert self.config.model.loss_type.lower() in ["sam", "medsam"], \
            ValueError(f"{self.config.model.loss_type} is not supported. Please use 'sam' or 'medsam'.")

        # set train metrics
        self.train_metrics = MetricCollection(dict(
            train_IoU=BinaryJaccardIndex(),
            train_F1=BinaryF1Score(),
            train_Dice=Dice(),
            train_loss=MeanMetric(),
            train_loss_dice=MeanMetric(),
        ))
        if self.config.model.loss_type.lower() == "sam":
            self.train_metrics.add_metrics(dict(
                train_loss_focal=MeanMetric(),
                train_loss_iou=MeanMetric(),
            ))
        elif self.config.model.loss_type.lower() == "medsam":
            self.train_metrics.add_metrics(dict(train_loss_ce=MeanMetric()))

        # set val metrics
        self.val_metrics = MetricCollection(dict(
            val_IoU=BinaryJaccardIndex(),
            val_F1=BinaryF1Score(),
            val_Dice=Dice(),
        ))

    def load_checkpoint(self, state_dict: Optional[OrderedDict] = None, revise_keys: List = [(r'^image_encoder.', r'image_encoder.backbone.')]) -> None:
        """"""
        def replace_state_dict_keys(state_dict, revise_keys):
            for p, r in revise_keys:
                state_dict = OrderedDict(
                    {re.sub(p, r, k) if re.search(p, k) and not re.search(r, k) else k: v for k, v in state_dict.items()}
                )
            return state_dict

        if state_dict:
            # state_dict from args.load_from
            state_dict = replace_state_dict_keys(state_dict, revise_keys)
            self.load_state_dict(state_dict)
        else:
            try:
                self.load_from_checkpoint(self.config.model.checkpoint)
            except:
                if str(self.config.model.checkpoint).startswith("http"):
                    state_dict = torch.hub.load_state_dict_from_url(str(self.config.model.checkpoint))
                else:
                    with open(self.config.model.checkpoint, "rb") as f:
                        state_dict = torch.load(f)
                state_dict = replace_state_dict_keys(state_dict, revise_keys)
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
                multimask_output=False, # when given multiple prompts. if there is single prompt True would be better.
            )

            pred_masks.append(low_res_masks)
            ious.append(iou_predictions)

        return pred_masks, ious

    def training_step(self, batch, batch_idx) -> Tensor:
        """Training step of SAM."""
        images = batch["images"]
        bboxes = batch["bboxes"]
        points = batch["points"]
        gt_masks = batch["gt_masks"]

        pred_masks, iou_predictions = self(images, bboxes, points)

        loss_dice = 0.
        if self.config.model.loss_type.lower() == "sam":
            loss_focal = 0.
            loss_iou = 0.
        elif self.config.model.loss_type.lower() == "medsam":
            loss_ce = 0.

        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
        for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
            pred_mask = self.postprocess_masks(pred_mask, images.shape[2:])
            pred_mask = pred_mask.sigmoid()
            self.train_metrics["train_IoU"].update(pred_mask, gt_mask)
            self.train_metrics["train_F1"].update(pred_mask, gt_mask)
            self.train_metrics["train_Dice"].update(pred_mask, gt_mask)
            pred_mask = pred_mask.flatten(1)
            gt_mask = gt_mask.flatten(1).float()

            # calculate losses
            loss_dice += self.calculate_dice_loss(pred_mask, gt_mask, num_masks)
            if self.config.model.loss_type.lower() == "sam":
                loss_focal += self.calculate_sigmoid_ce_focal_loss(pred_mask, gt_mask, num_masks)
                batch_iou = self.calculate_iou(pred_mask, gt_mask)
                loss_iou += F.mse_loss(iou_prediction, batch_iou.unsqueeze(1), reduction='sum') / num_masks

            elif self.config.model.loss_type.lower() == "medsam":
                loss_ce += self.calculate_sigmoid_ce_focal_loss(pred_mask, gt_mask, num_masks)

        if self.config.model.loss_type.lower() == "sam":
            loss = 20. * loss_focal + loss_dice + loss_iou
            self.train_metrics["train_loss_focal"].update(loss_focal)
            self.train_metrics["train_loss_iou"].update(loss_iou)

        elif self.config.model.loss_type.lower() == "medsam":
            loss = loss_dice + loss_ce
            self.train_metrics["train_loss_ce"].update(loss_ce)

        self.train_metrics["train_loss"].update(loss)
        self.train_metrics["train_loss_dice"].update(loss_dice)

        self.log_dict(self.train_metrics.compute(), prog_bar=True)

        return loss

    def training_epoch_end(self, outputs) -> None:
        for v in self.train_metrics.values():
            v.reset()

    def validation_step(self, batch, batch_idx) -> MetricCollection:
        """Validation step of SAM."""
        images = batch["images"]
        bboxes = batch["bboxes"]
        points = batch["points"]
        gt_masks = batch["gt_masks"]

        pred_masks, _ = self(images, bboxes, points)
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            pred_mask = self.postprocess_masks(pred_mask, images.shape[2:])
            pred_mask = pred_mask.sigmoid()
            for k, v in self.val_metrics.items():
                v.update(pred_mask, gt_mask)

        return self.val_metrics

    def validation_epoch_end(self, outputs) -> None:
        self.log_dict(self.val_metrics.compute(), on_epoch=True, prog_bar=True)
        for v in self.val_metrics.values():
            v.reset()

    def predict_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        """Predict step of SAM."""
        images = batch["images"]
        bboxes = batch["bboxes"]
        points = batch["points"]

        pred_masks, iou_predictions = self(images, bboxes, points)

        masks: List[Tensor] = []
        for i, pred_mask in enumerate(pred_masks):
            mask = self.postprocess_masks(pred_mask, images[i].shape[1:], batch["original_size"][i], is_predict=True)
            if not self.config.model.return_logits:
                mask = mask > self.config.model.mask_threshold
            else:
                mask = mask.sigmoid()
            masks.append(mask)

        return dict(masks=masks, iou_predictions=iou_predictions, path=batch["path"], labels=batch["labels"])

    def postprocess_masks(
        self,
        masks: Tensor,
        input_size: Tuple[int, ...],
        original_size: Optional[Tuple[int, ...]] = None,
        is_predict: bool = False
    ) -> Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (Tensor): Batched masks from the mask_decoder, in BxCxHxW format.
            input_size (tuple(int, int)): The size of the image input to the model, in (H, W) format. Used to remove padding.
            original_size (tuple(int, int)): The original size of the image before resizing for input to the model, in (H, W) format.
            is_predict (bool, optional): ...

        Returns:
          (Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(masks, input_size, mode="bilinear", align_corners=False)
        if is_predict:
            masks = masks[..., : input_size[0], : input_size[1]]
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks.squeeze(1)
    
    def configure_optimizers(self) -> optim:
        name = self.config.optimizer.pop("name")
        optimizer = getattr(optim, name)(self.parameters(), **self.config.optimizer)
        return optimizer
    
    def calculate_dice_loss(self, inputs: Tensor, targets: Tensor, num_masks: int) -> Tensor:
        """
        Compute the DICE loss, similar to generalized IOU for masks
        
        Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/modeling_maskformer.py#L269

        Args:
            inputs (Tensor):
                A tensor representing a mask.
            targets (Tensor):
                A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
                (0 for the negative class and 1 for the positive class).
            num_masks (int):
                The number of masks present in the current batch, used for normalization.

        Returns:
            Loss tensor
        """
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks
    
    def calculate_sigmoid_ce_focal_loss(
        self,
        inputs: Tensor,
        targets: Tensor,
        num_masks: int,
        alpha: float = 0.25,
        gamma: float = 2
    ) -> Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Referece: https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/modeling_maskformer.py#L300

        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
            targets (Tensor): A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
                (0 for the negative class and 1 for the positive class).
            num_masks (int): The number of masks present in the current batch, used for normalization.
            alpha (float, *optional*, defaults to 0.25): Weighting factor in range (0,1) to balance positive vs negative examples.
            gamma (float, *optional*, defaults to 2.0): Exponent of the modulating factor \\(1 - p_t\\) to balance easy vs hard examples.

        Returns:
            Loss tensor
        """
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        if self.config.model.loss_type.lower() == "sam":
            # focal loss for SAM loss
            p_t = inputs * targets + (1 - inputs) * (1 - targets)
            loss = loss * ((1 - p_t) ** gamma)
            if alpha >= 0:
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                loss = alpha_t * loss
        return loss.mean(1).sum() / num_masks
    
    def calculate_iou(self, inputs: Tensor, targets: Tensor, epsilon: float = 1e-7) -> Tensor:
        """"""
        pred_mask = (inputs >= 0.5).float()
        intersection = torch.sum(torch.mul(pred_mask, targets), dim=1)
        union = torch.sum(pred_mask, dim=1) + torch.sum(targets, dim=1) - intersection
        iou = intersection / (union + epsilon)
        return iou
