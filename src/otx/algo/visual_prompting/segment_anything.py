# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Segment Anything model for the OTX visual prompting."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import tv_tensors

from otx.algo.visual_prompting.decoders import SAMMaskDecoder
from otx.algo.visual_prompting.encoders import (SAMImageEncoder,
                                                SAMPromptEncoder)
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity)
from otx.core.model.entity.visual_prompting import OTXVisualPromptingModel

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SegmentAnything(nn.Module):
    def __init__(
        self,
        backbone: str,
        image_size: int = 1024,
        image_embedding_size: int = 64,
        embed_dim: int = 256,
        mask_in_chans: int = 16,
        num_multimask_outputs: int = 3,
        transformer_cfg: Dict[str, int] = dict(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        transformer_dim: int = 256,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

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
        
    def forward(
        self,
        images: Tensor,
        bboxes: List[Optional[Tensor]],
        # masks: Optional[List[Optional[Tensor]]] = None,
        points: Optional[List[Optional[Tuple[Tensor, Tensor]]]] = None, # TODO
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward method for SAM training/validation/prediction.

        Args:
            images (Tensor): Images with shape (B, C, H, W).
            bboxes (List[Tensor], optional): A Nx4 array given a box prompt to the model, in XYXY format.
            points (List[Tuple[Tensor, Tensor]], optional): Point coordinates and labels to embed.
                Point coordinates are BxNx2 arrays of point prompts to the model.
                Each point is in (X,Y) in pixels. Labels are BxN arrays of labels for the point prompts.
                1 indicates a foreground point and 0 indicates a background point.
            # masks (Optional[Tensor], optional): A low resolution mask input to the model, typically
            #     coming from a previous prediction iteration. Has form Bx1xHxW, where
            #     for SAM, H=W=256. Masks returned by a previous iteration of the
            #     predict method do not need further transformation.

        Returns:
            pred_masks (List[Tensor]): List with predicted masks with shape (B, 1, H, W).
            ious (List[Tensor]): List with IoU predictions with shape (N, 1).
        """
        image_embeddings = self.image_encoder(images)
        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, bboxes):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, # TODO
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
            ious.append(iou_predictions)

        if self.training:
            loss_dice = 0.0
            loss_focal = 0.0
            loss_iou = 0.0

            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            for i, (pred_mask, gt_mask, iou_prediction) in enumerate(zip(pred_masks, gt_masks, iou_predictions)):
                pred_mask = self.postprocess_masks(
                    pred_mask, self.image_size, batch["padding"][i], batch["original_size"][i]
                )
                pred_mask = pred_mask.sigmoid()
                pred_mask = pred_mask.flatten(1)
                gt_mask = gt_mask.flatten(1).float()

                # calculate losses
                loss_dice += self.calculate_dice_loss(pred_mask, gt_mask, num_masks)
                loss_focal += self.calculate_sigmoid_ce_focal_loss(pred_mask, gt_mask, num_masks)
                batch_iou = self.calculate_iou(pred_mask, gt_mask)
                loss_iou += F.mse_loss(iou_prediction, batch_iou.unsqueeze(1), reduction="sum") / num_masks

            loss = 20.0 * loss_focal + loss_dice + loss_iou
            return loss

        return pred_masks, ious
    
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
        self, inputs: Tensor, targets: Tensor, num_masks: int, alpha: float = 0.25, gamma: float = 2
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
        if self.config.model.loss_type.lower() == "sam":
            # focal loss for SAM loss
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
        iou = intersection / (union + epsilon)
        return iou
    
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

        prepadded_size = self.resize_longest_image_size(orig_size, input_size)
        masks = masks[..., : prepadded_size[0], : prepadded_size[1]]  # type: ignore

        orig_size = orig_size.to(torch.int64)
        h, w = orig_size[0], orig_size[1]
        masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
        return masks
    
    def resize_longest_image_size(self, input_image_size: Tensor, longest_side: int) -> Tensor:
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size


class OTXSegmentAnything(OTXVisualPromptingModel):
    def _create_model(self) -> nn.Module:
        """Create a PyTorch model for this class."""
        return SegmentAnything(backbone=self.config.backbone)

    def _customize_inputs(self, inputs: VisualPromptingBatchDataEntity) -> dict[str, Any]:
        """Customize the inputs for the model."""
        images = torch.stack(inputs.images, dim=0).to(dtype=torch.float32)
        return {
            "images": images,
            "bboxes": self._inspect_prompts(inputs.bboxes),
            # "points": self.inspect_prompts(inputs.points), # TODO
            # "masks": self._inspect_prompts(inputs.masks), # to be removed
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: VisualPromptingBatchDataEntity,
    ) -> VisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for you model."""
        if self.training:
            return {"loss": outputs}
        
        pred_masks, ious = outputs
        
        scores: list[torch.Tensor] = []
        labels: list[torch.LongTensor] = []
        masks: list[tv_tensors.Mask] = []
        for output in outputs:
            if not isinstance(output, DetDataSample):
                raise TypeError(output)

            scores.append(output.pred_instances.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    output.pred_instances.bboxes,
                    format="XYXY",
                    canvas_size=output.img_shape,
                ),
            )
            output_masks = tv_tensors.Mask(
                output.pred_instances.masks,
                dtype=torch.bool,
            )
            masks.append(output_masks)
            labels.append(output.pred_instances.labels)

        return InstanceSegBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks,
            polygons=[],
            labels=labels,
        )
    
    def _inspect_prompts(self, prompts: List[Union[tv_tensors.BoundingBoxes, tv_tensors.Mask]]) -> List[Optional[Union[tv_tensors.BoundingBoxes, tv_tensors.Mask]]]:
        """Inspect if given prompts are empty.
        
        If there are empty prompts (shape=0), they will be converted to None.
        """
        converted_prompts = [p if p.shape[0] > 0 else None for p in prompts]
        return converted_prompts
