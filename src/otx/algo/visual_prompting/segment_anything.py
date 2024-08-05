# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Segment Anything model for the OTX visual prompting."""

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torchvision import tv_tensors

from otx.algo.visual_prompting.decoders import SAMMaskDecoder
from otx.algo.visual_prompting.encoders import SAMImageEncoder, SAMPromptEncoder
from otx.core.data.entity.base import OTXBatchLossEntity, Points
from otx.core.data.entity.visual_prompting import VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity
from otx.core.metrics.visual_prompting import VisualPromptingMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.visual_prompting import OTXVisualPromptingModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes, NullLabelInfo

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable

DEFAULT_CONFIG_SEGMENT_ANYTHING: dict[str, dict[str, Any]] = {
    "tiny_vit": {
        "load_from": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
    },
    "vit_b": {
        "load_from": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
    "vit_l": {
        "load_from": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    },
    "vit_h": {
        "load_from": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
}


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
        use_stability_score: bool = False,
        return_single_mask: bool = False,
        return_extra_metrics: bool = False,
        stability_score_offset: float = 1.0,
    ) -> None:
        super().__init__()
        if transformer_cfg is None:
            transformer_cfg = {"depth": 2, "embedding_dim": 256, "mlp_dim": 2048, "num_heads": 8}

        self.mask_threshold = mask_threshold
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.use_stability_score = use_stability_score
        self.return_single_mask = return_single_mask
        self.return_extra_metrics = return_extra_metrics
        self.stability_score_offset = stability_score_offset

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
                if key in state_dict:
                    state_dict.pop(key)
            self.load_state_dict(state_dict)
        except ValueError as e:
            log.info(
                f"{e}: {load_from} is not desirable format for torch.hub.load_state_dict_from_url. "
                f"To manually load {load_from}, try to set it to trainer.checkpoint.",
            )

    def forward(self, *args, mode: str = "infer", **kwargs) -> Any:  # noqa: ANN401
        """Forward method for visual prompting task."""
        assert mode in ["finetuning", "learn", "infer"]  # noqa: S101
        if mode == "finetuning":
            return self.forward_train(*args, **kwargs)
        return self.forward_inference(*args, **kwargs)

    @torch.no_grad()
    def forward_inference(
        self,
        image_embeddings: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        mask_input: Tensor,
        has_mask_input: Tensor,
        ori_shape: Tensor,
    ) -> tuple[Tensor, ...]:
        """Forward method for SAM inference (export/deploy).

        Args:
            image_embeddings (Tensor): The image embedding with a batch index of length 1.
                If it is a zero tensor, the image embedding will be computed from the image.
            point_coords (Tensor): Coordinates of sparse input prompts,
                corresponding to both point inputs and box inputs.
                Boxes are encoded using two points, one for the top-left corner and one for the bottom-right corner.
                Coordinates must already be transformed to long-side 1024. Has a batch index of length 1.
            point_labels (Tensor): Labels for the sparse input prompts.
                0 is a negative input point, 1 is a positive input point,
                2 is a top-left box corner, 3 is a bottom-right box corner, and -1 is a padding point.
                If there is no box input, a single padding point with label -1 and
                coordinates (0.0, 0.0) should be concatenated.
            mask_input (Tensor): A mask input to the model with shape 1x1x256x256.
                This must be supplied even if there is no mask input. In this case, it can just be zeros.
            has_mask_input (Tensor): An indicator for the mask input.
                1 indicates a mask input, 0 indicates no mask input.
                This input has 1x1 shape due to supporting openvino input layout.
            ori_shape (Tensor): The size of the input image in (H,W) format, before any transformation.
                This input has 1x2 shape due to supporting openvino input layout.
        """
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        masks, scores = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.use_stability_score:
            scores = self.calculate_stability_score(
                masks,
                self.mask_threshold,
                self.stability_score_offset,
            )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

        upscaled_masks = self.postprocess_masks(masks, self.image_size, ori_shape)

        if self.return_extra_metrics:
            stability_scores = self.calculate_stability_score(
                upscaled_masks,
                self.mask_threshold,
                self.stability_score_offset,
            )
            areas = (upscaled_masks > self.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, scores, stability_scores, areas, masks

        return upscaled_masks, scores, masks

    def forward_train(
        self,
        images: tv_tensors.Image,
        ori_shapes: list[Tensor],
        bboxes: list[tv_tensors.BoundingBoxes | None],
        points: list[tuple[Points, Tensor] | None],
        gt_masks: list[tv_tensors.Mask] | None = None,
    ) -> Tensor | tuple[list[Tensor], list[Tensor]]:
        """Forward method for SAM training/validation/prediction.

        Args:
            images (tv_tensors.Image): Images with shape (B, C, H, W).
            ori_shapes (List[Tensor]): List of original shapes per image.
            bboxes (List[tv_tensors.BoundingBoxes], optional): A Nx4 array given a box prompt to the model,
                in XYXY format.
            points (List[Tuple[Points, Tensor]], optional): Point coordinates and labels to embed.
                Point coordinates are BxNx2 arrays of point prompts to the model.
                Each point is in (X,Y) in pixels. Labels are BxN arrays of labels for the point prompts.
                1 indicates a foreground point and 0 indicates a background point.
            gt_masks (List[tv_tensors.Mask], optional): Ground truth masks for loss calculation.

        Returns:
            (Tensor): Calculated loss values.
            (Tuple[List[Tensor], List[Tensor]]): Tuple of list with predicted masks with shape (B, 1, H, W)
                and List with IoU predictions with shape (N, 1).
        """
        image_embeddings = self.image_encoder(images)
        pred_masks = []
        ious = []
        for idx, embedding in enumerate(image_embeddings):
            low_res_masks, iou_predictions = [], []
            for prompt in [bboxes[idx], points[idx]]:
                if prompt is None:
                    continue

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=prompt if isinstance(prompt[0], Points) else None,
                    boxes=prompt if isinstance(prompt, tv_tensors.BoundingBoxes) else None,
                    masks=None,
                )
                _low_res_masks, _iou_predictions = self.mask_decoder(
                    image_embeddings=embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,  # when given multiple prompts. if there is single prompt True would be better. # noqa: E501
                )
                low_res_masks.append(_low_res_masks)
                iou_predictions.append(_iou_predictions)

            pred_masks.append(torch.cat(low_res_masks, dim=0))
            ious.append(torch.cat(iou_predictions, dim=0))

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
                loss_iou += F.mse_loss(iou, batch_iou.unsqueeze(1), reduction="sum") / num_masks

            loss = 20.0 * loss_focal + loss_dice + loss_iou

            return {"loss": loss, "loss_focal": loss_focal, "loss_dice": loss_dice, "loss_iou": loss_iou}

        post_processed_pred_masks: list[Tensor] = []
        for pred_mask, ori_shape in zip(pred_masks, ori_shapes):
            post_processed_pred_mask = self.postprocess_masks(pred_mask, self.image_size, ori_shape)
            post_processed_pred_masks.append(post_processed_pred_mask.squeeze(1).sigmoid())
        return post_processed_pred_masks, ious

    def _embed_points(self, point_coords: Tensor, point_labels: Tensor) -> Tensor:
        """Embed sparse input prompts.

        Args:
            point_coords (Tensor): Coordinates of sparse input prompts,
                corresponding to both point inputs and box inputs. Boxes are encoded using two points,
                one for the top-left corner and one for the bottom-right corner.
                Coordinates must already be transformed to long-side 1024. Has a batch index of length 1.
            point_labels (Tensor): Labels for the sparse input prompts.
                0 is a negative input point, 1 is a positive input point,
                2 is a top-left box corner, 3 is a bottom-right box corner, and -1 is a padding point.
                If there is no box input, a single padding point with label -1 and
                coordinates (0.0, 0.0) should be concatenated.

        Returns:
            point_embedding (Tensor): The embedded sparse input prompts.
        """
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.image_size
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)  # noqa: SLF001
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (point_labels == -1)

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[i].weight * (point_labels == i)

        return point_embedding

    def _embed_masks(self, input_mask: Tensor, has_mask_input: Tensor) -> Tensor:
        """Embed the mask input.

        Args:
            input_mask (Tensor): A mask input to the model with shape 1x1x256x256.
                This must be supplied even if there is no mask input. In this case, it can just be zeros.
            has_mask_input (Tensor): An indicator for the mask input.
                1 indicates a mask input, 0 indicates no mask input.

        Returns:
            mask_embedding (Tensor): The embedded mask input.
        """
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(input_mask)
        return mask_embedding + (1 - has_mask_input) * self.prompt_encoder.no_mask_embed.weight.reshape(
            1,
            -1,
            1,
            1,
        )

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

    @classmethod
    def postprocess_masks(cls, masks: Tensor, input_size: int, orig_size: Tensor) -> Tensor:
        """Postprocess the predicted masks.

        Args:
            masks (Tensor): A batch of predicted masks with shape Bx1xHxW.
            input_size (int): The size of the image input to the model, in (H, W) format.
                Used to remove padding.
            orig_size (Tensor): The original image size with shape Bx2.

        Returns:
            masks (Tensor): The postprocessed masks with shape Bx1xHxW.
        """
        orig_size = orig_size.squeeze()
        masks = F.interpolate(masks, size=(input_size, input_size), mode="bilinear", align_corners=False)

        prepadded_size = cls.get_prepadded_size(cls, orig_size, input_size)  # type: ignore[arg-type]
        masks = masks[..., : prepadded_size[0], : prepadded_size[1]]

        orig_size = orig_size.to(torch.int64)
        h, w = orig_size[0], orig_size[1]
        return F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)

    def get_prepadded_size(self, input_image_size: Tensor, longest_side: int) -> Tensor:
        """Get pre-padded size."""
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        return torch.floor(transformed_size + 0.5).to(torch.int64)

    def calculate_stability_score(self, masks: Tensor, mask_threshold: float, threshold_offset: float = 1.0) -> Tensor:
        """Computes the stability score for a batch of masks.

        The stability score is the IoU between the binary masks obtained
        by thresholding the predicted mask logits at high and low values.

        Args:
            masks (Tensor): A batch of predicted masks with shape BxHxW.
            mask_threshold (float): The threshold used to binarize the masks.
            threshold_offset (float, optional): The offset used to compute the stability score.

        Returns:
            stability_scores (Tensor): The stability scores for the batch of masks.
        """
        # One mask is always contained inside the other.
        # Save memory by preventing unnecessary cast to torch.int64
        intersections = (
            (masks > (mask_threshold + threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
        )
        unions = (masks > (mask_threshold - threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
        return intersections / unions

    def select_masks(self, masks: Tensor, iou_preds: Tensor, num_points: int) -> tuple[Tensor, Tensor]:
        """Selects the best mask from a batch of masks.

        Args:
            masks (Tensor): A batch of predicted masks with shape BxMxHxW.
            iou_preds (Tensor): A batch of predicted IoU scores with shape BxM.
            num_points (int): The number of points in the input.

        Returns:
            masks (Tensor): The selected masks with shape Bx1xHxW.
            iou_preds (Tensor): The selected IoU scores with shape Bx1.
        """
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = torch.tensor([[1000] + [0] * (self.mask_decoder.num_mask_tokens - 1)]).to(iou_preds.device)
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds


class OTXSegmentAnything(OTXVisualPromptingModel):
    """Visual Prompting model."""

    def __init__(
        self,
        backbone: Literal["tiny_vit", "vit_b"],
        label_info: LabelInfoTypes = NullLabelInfo(),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = VisualPromptingMetricCallable,
        torch_compile: bool = False,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
        use_stability_score: bool = False,
        return_single_mask: bool = True,
        return_extra_metrics: bool = False,
        stability_score_offset: float = 1.0,
    ) -> None:
        self.config = {
            "backbone": backbone,
            "freeze_image_encoder": freeze_image_encoder,
            "freeze_prompt_encoder": freeze_prompt_encoder,
            "freeze_mask_decoder": freeze_mask_decoder,
            "use_stability_score": use_stability_score,
            "return_single_mask": return_single_mask,
            "return_extra_metrics": return_extra_metrics,
            "stability_score_offset": stability_score_offset,
            **DEFAULT_CONFIG_SEGMENT_ANYTHING[backbone],
        }
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        """Create a PyTorch model for this class."""
        return SegmentAnything(**self.config)

    def _customize_inputs(self, inputs: VisualPromptingBatchDataEntity) -> dict[str, Any]:  # type: ignore[override]
        """Customize the inputs for the model."""
        images = tv_tensors.wrap(torch.stack(inputs.images, dim=0).to(dtype=torch.float32), like=inputs.images[0])
        return {
            "mode": "finetuning",
            "images": images,
            "ori_shapes": [torch.tensor(info.ori_shape) for info in inputs.imgs_info],
            "gt_masks": inputs.masks,
            "bboxes": self._inspect_prompts(inputs.bboxes),
            "points": [
                (
                    (tv_tensors.wrap(point.unsqueeze(1), like=point), torch.ones(len(point), 1, device=point.device))
                    if point is not None
                    else None
                )
                for point in self._inspect_prompts(inputs.points)
            ],
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: VisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> VisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for model."""
        if self.training:
            return outputs

        masks: list[tv_tensors.Mask] = []
        scores: list[torch.Tensor] = []
        for mask, score in zip(*outputs):
            masks.append(tv_tensors.Mask(mask, dtype=torch.float32))
            scores.append(score)

        return VisualPromptingBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            masks=masks,
            polygons=[],
            points=[],
            bboxes=[],
            labels=[torch.cat(list(labels.values())) for labels in inputs.labels],
        )

    def _inspect_prompts(self, prompts: list[tv_tensors.TVTensor]) -> list[tv_tensors.TVTensor | None]:
        """Inspect if given prompts are empty.

        If there are empty prompts (shape=0), they will be converted to None.
        """
        return [None if p is None else None if p.shape[0] == 0 else p for p in prompts]
