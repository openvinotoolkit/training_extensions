# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Segment Anything model for the OTX visual prompting."""

from __future__ import annotations

import logging as log
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torchvision import tv_tensors

from otx.algo.visual_prompting.decoders import SAMMaskDecoder
from otx.algo.visual_prompting.encoders import SAMImageEncoder, SAMPromptEncoder
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.visual_prompting import VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity
from otx.core.model.entity.visual_prompting import OTXVisualPromptingModel


class SegmentAnything(nn.Module):
    """Visual prompting model class for Segment Anything."""

    def __init__(
        self,
        backbone: str,
        load_from: str | None = None,
        mask_threshold: float = 0.0,
        image_size: int = 1024,
        image_embedding_size: int = 64,
        embed_dim: int = 256,
        mask_in_chans: int = 16,
        num_multimask_outputs: int = 3,
        transformer_cfg: dict[str, int] | None = None,
        transformer_dim: int = 256,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
    ) -> None:
        super().__init__()
        if transformer_cfg is None:
            transformer_cfg = {"depth": 2, "embedding_dim": 256, "mlp_dim": 2048, "num_heads": 8}

        self.mask_threshold = mask_threshold
        self.image_size = image_size

        self.image_encoder = SAMImageEncoder(backbone=backbone)
        self.prompt_encoder = SAMPromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=mask_in_chans,
        )
        self.mask_decoder = SAMMaskDecoder(
            num_multimask_outputs=num_multimask_outputs,
            transformer_cfg=transformer_cfg,
            transformer_dim=transformer_dim,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
        )

        self.load_checkpoint(load_from=load_from)
        self.freeze_networks(freeze_image_encoder, freeze_prompt_encoder, freeze_mask_decoder)

    def freeze_networks(
        self,
        freeze_image_encoder: bool,
        freeze_prompt_encoder: bool,
        freeze_mask_decoder: bool,
    ) -> None:
        """Freeze networks depending on config."""
        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        if freeze_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

        if freeze_mask_decoder:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False

    def load_checkpoint(
        self,
        load_from: str | None,
    ) -> None:
        """Load checkpoint for SAM.

        Args:
            load_from (Optional[str], optional): Checkpoint path for SAM. Defaults to None.
        """
        try:
            state_dict = torch.hub.load_state_dict_from_url(str(load_from))
            for key in [
                "image_encoder.norm_head.weight",
                "image_encoder.norm_head.bias",
                "image_encoder.head.weight",
                "image_encoder.head.bias",
            ]:
                state_dict.pop(key)
            self.load_state_dict(state_dict)
        except ValueError as e:
            log.info(
                f"{e}: {load_from} is not desirable format for torch.hub.load_state_dict_from_url. "
                f"To manually load {load_from}, try to set it to trainer.checkpoint.",
            )

    def forward(
        self,
        images: Tensor,
        ori_shapes: list[Tensor],
        bboxes: list[Tensor | None] | None = None,
        points: list[tuple[Tensor, Tensor] | None] | None = None,  # TODO(sungchul): enable point prompts # noqa: TD003
        gt_masks: list[Tensor] | None = None,
    ) -> Tensor | tuple[list[Tensor], list[Tensor]]:
        """Forward method for SAM training/validation/prediction.

        Args:
            images (Tensor): Images with shape (B, C, H, W).
            ori_shapes (List[Tensor]): List of original shapes per image.
            bboxes (List[Tensor], optional): A Nx4 array given a box prompt to the model, in XYXY format.
            points (List[Tuple[Tensor, Tensor]], optional): Point coordinates and labels to embed.
                Point coordinates are BxNx2 arrays of point prompts to the model.
                Each point is in (X,Y) in pixels. Labels are BxN arrays of labels for the point prompts.
                1 indicates a foreground point and 0 indicates a background point.
            # masks (Optional[Tensor], optional): A low resolution mask input to the model, typically
            #     coming from a previous prediction iteration. Has form Bx1xHxW, where
            #     for SAM, H=W=256. Masks returned by a previous iteration of the
            #     predict method do not need further transformation.
            gt_masks (List[Tensor], optional): Ground truth masks for loss calculation.

        Returns:
            (Tensor): Calculated loss values.
            (Tuple[List[Tensor], List[Tensor]]): Tuple of list with predicted masks with shape (B, 1, H, W)
                and List with IoU predictions with shape (N, 1).
        """
        image_embeddings = self.image_encoder(images)
        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, bboxes):  # type: ignore[arg-type]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,  # TODO(sungchul): enable point prompts # noqa: TD003
                boxes=bbox,
                masks=None,
            )

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,  # when given multiple prompts. if there is single prompt True would be better.
            )

            pred_masks.append(low_res_masks)
            ious.append(iou_predictions.squeeze())

        if self.training:
            loss_dice = 0.0
            loss_focal = 0.0
            loss_iou = 0.0

            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            for pred_mask, gt_mask, iou, ori_shape in zip(pred_masks, gt_masks, ious, ori_shapes):  # type: ignore[arg-type]
                post_processed_pred_mask = self.postprocess_masks(pred_mask, self.image_size, ori_shape)
                post_processed_pred_mask = post_processed_pred_mask.sigmoid()
                post_processed_pred_mask = post_processed_pred_mask.flatten(1)
                flatten_gt_mask = gt_mask.flatten(1).float()

                # calculate losses
                loss_dice += self.calculate_dice_loss(post_processed_pred_mask, flatten_gt_mask, num_masks)
                loss_focal += self.calculate_sigmoid_ce_focal_loss(post_processed_pred_mask, flatten_gt_mask, num_masks)
                batch_iou = self.calculate_iou(post_processed_pred_mask, flatten_gt_mask)
                loss_iou += F.mse_loss(iou, batch_iou, reduction="sum") / num_masks

            loss = 20.0 * loss_focal + loss_dice + loss_iou

            return {"loss": loss, "loss_focal": loss_focal, "loss_dice": loss_dice, "loss_iou": loss_iou}

        post_processed_pred_masks: list[Tensor] = []
        for pred_mask, ori_shape in zip(pred_masks, ori_shapes):
            post_processed_pred_mask = self.postprocess_masks(pred_mask, self.image_size, ori_shape)
            post_processed_pred_masks.append(post_processed_pred_mask.squeeze().sigmoid())
        return post_processed_pred_masks, ious

    def calculate_dice_loss(self, inputs: Tensor, targets: Tensor, num_masks: int) -> Tensor:
        """Compute the DICE loss, similar to generalized IOU for masks.

        Args:
            inputs (Tensor): A tensor representing a mask.
            targets (Tensor): A tensor with the same shape as inputs. Stores the binary classification labels
                for each element in inputs (0 for the negative class and 1 for the positive class).
            num_masks (int): The number of masks present in the current batch, used for normalization.

        Returns:
            Tensor: The DICE loss.
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
        gamma: float = 2,
    ) -> Tensor:
        r"""Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002. # noqa: D301.

        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
            targets (Tensor): A tensor with the same shape as inputs. Stores the binary classification labels
                for each element in inputs (0 for the negative class and 1 for the positive class).
            num_masks (int): The number of masks present in the current batch, used for normalization.
            alpha (float, *optional*, defaults to 0.25): Weighting factor in range (0,1)
                to balance positive vs negative examples.
            gamma (float, *optional*, defaults to 2.0): Exponent of the modulating factor \\(1 - p_t\\)
                to balance easy vs hard examples.

        Returns:
            Tensor: The focal loss.
        """
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean(1).sum() / num_masks

    def calculate_iou(self, inputs: Tensor, targets: Tensor, epsilon: float = 1e-7) -> Tensor:
        """Calculate the intersection over union (IOU) between the predicted mask and the ground truth mask.

        Args:
            inputs (Tensor): A tensor representing a mask.
            targets (Tensor): A tensor with the same shape as inputs. Stores the binary classification labels
                for each element in inputs (0 for the negative class and 1 for the positive class).
            epsilon (float, *optional*, defaults to 1e-7): A small value to prevent division by zero.

        Returns:
            Tensor: The IOU between the predicted mask and the ground truth mask.
        """
        pred_mask = (inputs >= 0.5).float()
        intersection = torch.sum(torch.mul(pred_mask, targets), dim=1)
        union = torch.sum(pred_mask, dim=1) + torch.sum(targets, dim=1) - intersection
        return intersection / (union + epsilon)

    def postprocess_masks(self, masks: Tensor, input_size: int, orig_size: Tensor) -> Tensor:
        """Postprocess the predicted masks.

        Args:
            masks (Tensor): A batch of predicted masks with shape Bx1xHxW.
            input_size (int): The size of the image input to the model, in (H, W) format.
                Used to remove padding.
            orig_size (Tensor): The original image size with shape Bx2.

        Returns:
            masks (Tensor): The postprocessed masks with shape Bx1xHxW.
        """
        masks = F.interpolate(masks, size=(input_size, input_size), mode="bilinear", align_corners=False)

        prepadded_size = self.get_prepadded_size(orig_size, input_size)
        masks = masks[..., : prepadded_size[0], : prepadded_size[1]]

        orig_size = orig_size.to(torch.int64)
        h, w = orig_size[0], orig_size[1]
        return F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)

    def get_prepadded_size(self, input_image_size: Tensor, longest_side: int) -> Tensor:
        """Get pre-padded size."""
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        return torch.floor(transformed_size + 0.5).to(torch.int64)


class OTXSegmentAnything(OTXVisualPromptingModel):
    """Visual Prompting model."""

    def _create_model(self) -> nn.Module:
        """Create a PyTorch model for this class."""
        return SegmentAnything(**self.config)

    def _customize_inputs(self, inputs: VisualPromptingBatchDataEntity) -> dict[str, Any]:
        """Customize the inputs for the model."""
        images = torch.stack(inputs.images, dim=0).to(dtype=torch.float32)
        return {
            "images": images,
            "ori_shapes": [torch.tensor(info.ori_shape) for info in inputs.imgs_info],
            "bboxes": self._inspect_prompts(inputs.bboxes),
            # "points": self.inspect_prompts(inputs.points), # TODO(sungchul): enable point prompts # noqa: TD003
            "gt_masks": inputs.masks,
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: VisualPromptingBatchDataEntity,
    ) -> VisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for you model."""
        if self.training:
            return outputs

        masks: list[tv_tensors.Mask] = []
        scores: list[torch.Tensor] = []
        labels: list[torch.LongTensor] = inputs.labels
        for mask, score in zip(*outputs):
            masks.append(tv_tensors.Mask(mask, dtype=torch.float32))
            scores.append(score)

        return VisualPromptingBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=[],
            masks=masks,
            polygons=[],
            labels=labels,
        )

    def _inspect_prompts(self, prompts: list[tv_tensors.BoundingBoxes]) -> list[tv_tensors.BoundingBoxes | None]:
        """Inspect if given prompts are empty.

        If there are empty prompts (shape=0), they will be converted to None.
        """
        return [p if p.shape[0] > 0 else None for p in prompts]
