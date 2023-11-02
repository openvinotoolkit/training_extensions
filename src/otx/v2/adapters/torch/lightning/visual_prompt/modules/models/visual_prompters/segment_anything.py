"""SAM module for visual prompting.

paper: https://arxiv.org/abs/2304.02643
reference: https://github.com/facebookresearch/segment-anything
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from torch import Tensor, optim
from torch.nn import functional
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, Dice

from otx.v2.adapters.torch.lightning.model import BaseOTXLightningModel
from otx.v2.adapters.torch.lightning.visual_prompt.modules.models.decoders import SAMMaskDecoder
from otx.v2.adapters.torch.lightning.visual_prompt.modules.models.encoders import SAMImageEncoder, SAMPromptEncoder

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from pytorch_lightning.trainer.connectors.accelerator_connector import (
        _PRECISION_INPUT,
    )

CKPT_PATHS = {
    "tiny_vit": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


class SegmentAnything(BaseOTXLightningModel, LightningModule):
    """SAM predicts object masks from an image and input prompts."""

    def __init__(self, config: DictConfig, state_dict: OrderedDict | None = None) -> None:
        """Initializes the SegmentAnything model.

        Args:
            config (DictConfig): The configuration for the model.
            state_dict (OrderedDict | None): The state dictionary for the model, if available.

        Returns:
            None
        """
        super().__init__()
        self.save_hyperparameters(ignore="state_dict")
        self.config = config

        self.set_models()
        self.freeze_networks()
        self.set_metrics()
        self.load_checkpoint(state_dict=state_dict)

    def set_models(self) -> None:
        """Set models for SAM."""
        # Depending on backbone, image_embedding_size can be changed.
        if "vit" in self.config.model.backbone:
            patch_size = 16
            self.image_embedding_size = self.config.model.image_size // patch_size
        else:
            msg = f"{self.config.model.backbone} for image encoder of SAM is not implemented yet. Use vit_b, l, or h."
            raise NotImplementedError(
                msg,
            )

        self.image_encoder = SAMImageEncoder(self.config.model)
        self.prompt_encoder = SAMPromptEncoder(
            embed_dim=256,
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=(self.config.model.image_size, self.config.model.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = SAMMaskDecoder(
            num_multimask_outputs=3,
            transformer_cfg={"depth": 2, "embedding_dim": 256, "mlp_dim": 2048, "num_heads": 8},
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

    def freeze_networks(self) -> None:
        """Freeze networks depending on config."""
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
        """Set metrics for SAM."""
        # set train metrics
        self.train_metrics = MetricCollection(
            {
                "train_IoU": BinaryJaccardIndex(),
                "train_F1": BinaryF1Score(),
                "train_Dice": Dice(),
                "train_loss": MeanMetric(),
                "train_loss_dice": MeanMetric(),
            },
        )
        if self.config.model.loss_type.lower() == "sam":
            self.train_metrics.add_metrics(
                {
                    "train_loss_focal": MeanMetric(),
                    "train_loss_iou": MeanMetric(),
                },
            )
        elif self.config.model.loss_type.lower() == "medsam":
            self.train_metrics.add_metrics({"train_loss_ce": MeanMetric()})

        # set val metrics
        self.val_metrics = MetricCollection(
            {
                "val_IoU": BinaryJaccardIndex(),
                "val_F1": BinaryF1Score(),
                "val_Dice": Dice(),
            },
        )

        # set test metrics
        self.test_metrics = MetricCollection(
            {
                "test_IoU": BinaryJaccardIndex(),
                "test_F1": BinaryF1Score(),
                "test_Dice": Dice(),
            },
        )

    def load_checkpoint(
        self,
        state_dict: OrderedDict | None = None,
        revise_keys: list | None = None,
    ) -> None:
        """Load checkpoint for SAM.

        Args:
            state_dict (Optional[OrderedDict], optional): State dict of SAM. Defaults to None.
            revise_keys (List, optional): List of tuples of regex patterns to revise keys of state_dict.
                Defaults to [(r'^image_encoder.', r'image_encoder.backbone.')].
        """
        if revise_keys is None:
            revise_keys = [("^image_encoder.", "image_encoder.backbone.")]

        def replace_state_dict_keys(state_dict: OrderedDict, revise_keys: list) -> OrderedDict:
            for p, r in revise_keys:
                state_dict = OrderedDict(
                    {
                        re.sub(p, r, k) if re.search(p, k) and not re.search(r, k) else k: v
                        for k, v in state_dict.items()
                    },
                )
            return state_dict

        if revise_keys is None:
            revise_keys = [(r"^image_encoder.", r"image_encoder.backbone.")]
        if state_dict:
            # state_dict from args.load_from
            state_dict = replace_state_dict_keys(state_dict, revise_keys)
            self.load_state_dict(state_dict)
        elif self.config.model.checkpoint:
            if str(self.config.model.checkpoint).endswith(".ckpt"):
                # load lightning checkpoint
                self.load_from_checkpoint(self.config.model.checkpoint)
            else:
                if str(self.config.model.checkpoint).startswith("http"):
                    # get checkpoint from url
                    _state_dict = torch.hub.load_state_dict_from_url(str(self.config.model.checkpoint))
                else:
                    # load checkpoint from local
                    with Path(self.config.model.checkpoint).open("rb") as f:
                        _state_dict = torch.load(f)
                state_dict = replace_state_dict_keys(_state_dict, revise_keys)
                self.load_state_dict(state_dict, strict=False)

    ##########################################################
    #     forward for inference (export/deploy/optimize)     #
    ##########################################################
    @torch.no_grad()
    def forward(
        self,
        image_embeddings: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        mask_input: Tensor,
        has_mask_input: Tensor,
        # orig_size: Tensor,
    ) -> tuple:
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
            orig_size (Tensor): The size of the input image in (H,W) format, before any transformation.
                This input has 1x2 shape due to supporting openvino input layout.

        Returns:
            tuple: Tuple of scores and masks.
        """
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        masks, scores = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.config.model.use_stability_score:
            scores = self.calculate_stability_score(
                masks,
                self.config.model.mask_threshold,
                self.config.model.stability_score_offset,
            )

        if self.config.model.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

        return scores, masks

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
        point_coords = point_coords / self.config.model.image_size
        point_embedding = self.prompt_encoder.pe_layer.pe_encoding(point_coords)
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

    def mask_postprocessing(self, masks: Tensor, orig_size: Tensor) -> Tensor:
        """Postprocesses the predicted masks.

        Args:
            masks (Tensor): A batch of predicted masks with shape Bx1xHxW.
            orig_size (Tensor): The original image size with shape Bx2.

        Returns:
            masks (Tensor): The postprocessed masks with shape Bx1xHxW.
        """
        masks = functional.interpolate(
            masks,
            size=(self.config.model.image_size, self.config.model.image_size),
            mode="bilinear",
            align_corners=False,
        )

        prepadded_size = self.resize_longest_image_size(orig_size, self.config.model.image_size).to(torch.int64)
        masks = masks[..., : prepadded_size[0], : prepadded_size[1]]

        orig_size = orig_size.to(torch.int64)
        h, w = orig_size[0], orig_size[1]
        return functional.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)

    def resize_longest_image_size(self, input_image_size: Tensor, longest_side: int) -> Tensor:
        """Resizes the longest side of the image to the given size.

        Args:
            input_image_size (Tensor): The original image size with shape Bx2.
            longest_side (int): The size of the longest side.

        Returns:
            transformed_size (Tensor): The transformed image size with shape Bx2.
        """
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        return torch.floor(transformed_size + 0.5).to(torch.int64)

    ######################################################
    #     forward for training/validation/prediction     #
    ######################################################
    def forward_train(
        self,
        images: Tensor,
        bboxes: list[Tensor],
        points: tuple[Tensor, Tensor] | None = None,
        masks: Tensor | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Forward method for SAM training/validation/prediction.

        Args:
            images (Tensor): Images with shape (B, C, H, W).
            bboxes (List[Tensor]): A Nx4 array given a box prompt to the model, in XYXY format.
            points (Tuple[Tensor, Tensor], optional): Point coordinates and labels to embed.
                Point coordinates are BxNx2 arrays of point prompts to the model.
                Each point is in (X,Y) in pixels. Labels are BxN arrays of labels for the point prompts.
                1 indicates a foreground point and 0 indicates a background point.
            masks (Optional[Tensor], optional): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form Bx1xHxW, where
                for SAM, H=W=256. Masks returned by a previous iteration of the
                predict method do not need further transformation.

        Returns:
            pred_masks (List[Tensor]): List with predicted masks with shape (B, 1, H, W).
            ious (List[Tensor]): List with IoU predictions with shape (N, 1).
        """
        image_embeddings = self.image_encoder(images)
        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, bboxes):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=bbox,
                masks=masks,
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

        return pred_masks, ious

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Training step for SAM.

        Args:
            batch (Dict): Batch data.
            batch_idx (int): Batch index.

        Returns:
            loss (Tensor): Loss tensor.
        """
        _ = batch_idx
        images = batch["images"]
        bboxes = batch["bboxes"]
        points = batch["points"]
        gt_masks = batch["gt_masks"]

        pred_masks, iou_predictions = self.forward_train(images, bboxes, points)

        loss_dice = 0.0
        if self.config.model.loss_type.lower() == "sam":
            loss_focal = 0.0
            loss_iou = 0.0
        elif self.config.model.loss_type.lower() == "medsam":
            loss_ce = 0.0

        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
        for i, (pred_mask, gt_mask, iou_prediction) in enumerate(zip(pred_masks, gt_masks, iou_predictions)):
            _pred_mask = self.postprocess_masks(
                pred_mask,
                images.shape[2:],
                batch["padding"][i],
                batch["original_size"][i],
            )
            _pred_mask = _pred_mask.sigmoid()
            self.train_metrics["train_IoU"].update(_pred_mask, gt_mask)
            self.train_metrics["train_F1"].update(_pred_mask, gt_mask)
            self.train_metrics["train_Dice"].update(_pred_mask, gt_mask)
            _pred_mask = _pred_mask.flatten(1)
            _gt_mask = gt_mask.flatten(1).float()

            # calculate losses
            loss_dice += self.calculate_dice_loss(_pred_mask, _gt_mask, num_masks)
            if self.config.model.loss_type.lower() == "sam":
                loss_focal += self.calculate_sigmoid_ce_focal_loss(_pred_mask, _gt_mask, num_masks)
                batch_iou = self.calculate_iou(_pred_mask, _gt_mask)
                loss_iou += functional.mse_loss(iou_prediction, batch_iou.unsqueeze(1), reduction="sum") / num_masks

            elif self.config.model.loss_type.lower() == "medsam":
                loss_ce += self.calculate_sigmoid_ce_focal_loss(_pred_mask, _gt_mask, num_masks)

        if self.config.model.loss_type.lower() == "sam":
            loss = 20.0 * loss_focal + loss_dice + loss_iou
            self.train_metrics["train_loss_focal"].update(loss_focal)
            self.train_metrics["train_loss_iou"].update(loss_iou)

        elif self.config.model.loss_type.lower() == "medsam":
            loss = loss_dice + loss_ce
            self.train_metrics["train_loss_ce"].update(loss_ce)

        self.train_metrics["train_loss"].update(loss)
        self.train_metrics["train_loss_dice"].update(loss_dice)

        self.log_dict(self.train_metrics.compute(), prog_bar=True)

        return loss

    def training_epoch_end(self, outputs: Tensor) -> None:
        """Training epoch end for SAM."""
        _ = outputs
        for v in self.train_metrics.values():
            v.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> MetricCollection:
        """Validation step of SAM.

        Args:
            batch (Dict): Batch data.
            batch_idx (int): Batch index.

        Returns:
            val_metrics (MetricCollection): Validation metrics.
        """
        _ = batch_idx
        images = batch["images"]
        bboxes = batch["bboxes"]
        points = batch["points"]
        gt_masks = batch["gt_masks"]

        pred_masks, _ = self.forward_train(images, bboxes, points)
        for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
            _pred_mask = self.postprocess_masks(
                pred_mask,
                images.shape[2:],
                batch["padding"][i],
                batch["original_size"][i],
            )
            _pred_mask = _pred_mask.sigmoid()
            for v in self.val_metrics.values():
                v.update(_pred_mask, gt_mask)

        return self.val_metrics

    def validation_epoch_end(self, outputs: Tensor) -> None:
        """Validation epoch end for SAM."""
        _ = outputs
        self.log_dict(self.val_metrics.compute(), on_epoch=True, prog_bar=True)
        for v in self.val_metrics.values():
            v.reset()

    def predict_step(self, batch: dict, batch_idx: int) -> dict[str, Tensor]:
        """Predict step of SAM.

        Args:
            batch (Dict): Batch data.
            batch_idx (int): Batch index.

        Returns:
            Dict[str, Tensor]: Predicted masks, IoU predictions, image paths, and labels.
        """
        _ = batch_idx
        images = batch["images"]
        bboxes = batch["bboxes"]
        points = batch["points"]

        pred_masks, iou_predictions = self.forward_train(images, bboxes, points)

        masks: list[Tensor] = []
        for i, pred_mask in enumerate(pred_masks):
            mask = self.postprocess_masks(pred_mask, images.shape[2:], batch["padding"][i], batch["original_size"][i])
            if not self.config.model.return_logits:
                mask = (mask > self.config.model.mask_threshold).to(mask.dtype)
            else:
                mask = mask.sigmoid()
            masks.append(mask)

        return {"masks": masks, "iou_predictions": iou_predictions, "path": batch["path"], "labels": batch["labels"]}

    def test_step(self, batch: dict, batch_idx: int) -> dict[str, Tensor]:
        """Test step of SAM.

        Args:
            batch (Dict): Batch data.
            batch_idx (int): Batch index.

        Returns:
            Dict[str, Tensor]: Predicted masks, IoU predictions, image paths, and labels.
        """
        pred = self.predict_step(batch, batch_idx)
        for _, (pred_mask, gt_mask) in enumerate(zip(pred["masks"], batch["gt_masks"])):
            for t in self.test_metrics.values():
                t.update(pred_mask, gt_mask)
        return self.test_metrics

    def test_epoch_end(self, outputs: Tensor) -> None:
        """Test epoch end for SAM."""
        _ = outputs
        self.log_dict(self.test_metrics.compute(), on_epoch=True, prog_bar=True)
        for t in self.test_metrics.values():
            t.reset()

    def postprocess_masks(
        self,
        masks: Tensor,
        input_size: tuple[int, int],
        padding: tuple[int, ...],
        original_size: tuple[int, int],
    ) -> Tensor:
        """Remove padding and upscale masks to the original image size.

        Args:
            masks (Tensor): Predicted masks from the mask_decoder with (N, 1, H/downsized_ratio, W/downsized_ratio).
            input_size (tuple(int, int)): The size of the image input to the model, in (H, W) format.
                Used to remove padding.
            padding (tuple(int, int, int, int), optional): The padding applied to the image before input to the model,
                in (left, top, right, bottom) format.
            original_size (tuple(int, int)): The original size of the image before resizing for input to the model,
                in (H, W) format.

        Returns:
            Tensor: Postprocessed masks in NxHxW format, where (H, W) is given by original_size.
        """
        masks = functional.interpolate(masks, input_size, mode="bilinear", align_corners=False)
        masks = masks[..., : input_size[0] - padding[3], : input_size[1] - padding[2]]
        masks = functional.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks.squeeze(1)

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure the optimizer for SAM.

        Returns:
            optim: Optimizer.
        """
        name = self.config.optimizer.pop("name")
        return getattr(optim, name)(self.parameters(), **self.config.optimizer)

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
        r"""Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

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
        loss = functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
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
        return intersection / (union + epsilon)

    def export(
        self,
        export_dir: str | Path,
        export_type: str = "OPENVINO",
        precision: _PRECISION_INPUT | None = None,
    ) -> dict:
        """Exports the model to the specified format and precision.

        Args:
            export_dir (str | Path): The directory to export the model to.
            export_type (str, optional): The type of export to perform. Defaults to "OPENVINO".
            precision (_PRECISION_INPUT | None, optional): The precision to use for the export. Defaults to None.

        Returns:
            dict: A dictionary containing the paths to the exported models.
        """
        onnx_dir = Path(export_dir) / "onnx"
        onnx_dir.mkdir(exist_ok=True, parents=True)
        results: dict = {"outputs": {}}
        # 1) visual_prompting_image_encoder
        height = width = self.config.model.image_size
        dummy_inputs = {"images": torch.randn(1, 3, height, width, dtype=torch.float)}
        torch.onnx.export(
            model=self.image_encoder,
            args=tuple(dummy_inputs.values()),
            f=onnx_dir / "sam_encoder.onnx",
            export_params=True,
            verbose=False,
            opset_version=13,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=["image_embeddings"],
            dynamic_axes=None,
        )
        results["outputs"]["onnx"] = {}
        results["outputs"]["onnx"]["encoder"] = str(onnx_dir / "sam_encoder.onnx")

        # 2) SAM without backbone
        embed_dim = self.prompt_encoder.embed_dim
        embed_size = self.prompt_encoder.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]
        dynamic_axes = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }
        dummy_inputs = {
            "image_embeddings": torch.zeros(1, embed_dim, *embed_size, dtype=torch.float),
            "point_coords": torch.randint(low=0, high=1024, size=(1, 2, 2), dtype=torch.float),
            "point_labels": torch.randint(low=0, high=4, size=(1, 2), dtype=torch.float),
            "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
            "has_mask_input": torch.tensor([[1]], dtype=torch.float),
        }
        torch.onnx.export(
            model=self,
            args=tuple(dummy_inputs.values()),
            f=onnx_dir / "sam.onnx",
            export_params=True,
            verbose=False,
            opset_version=13,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=["iou_predictions", "low_res_masks"],
            dynamic_axes=dynamic_axes,
        )

        results["outputs"]["onnx"]["sam"] = str(onnx_dir / "sam.onnx")

        if export_type.upper() == "OPENVINO":
            ir_dir =  Path(export_dir) / "openvino"
            ir_dir.mkdir(exist_ok=True, parents=True)
            results["outputs"]["openvino"] = {}
            from subprocess import run
            for onnx_key, model_path in results["outputs"]["onnx"].items():
                optimize_command = [
                    "mo",
                    "--input_model",
                    model_path,
                    "--output_dir",
                    ir_dir,
                    "--model_name",
                    onnx_key,
                ]
                if onnx_key == "encoder":
                    dataset_config = self.config.get("dataset", {})
                    normalize = dataset_config.get("normalize", {})
                    mean = normalize.get("mean", [123.675, 116.28, 103.53])
                    std = normalize.get("std", [58.395, 57.12, 57.375])
                    optimize_command += [
                        "--mean_values",
                        str(mean).replace(", ", ","),
                        "--scale_values",
                        str(std).replace(", ", ","),
                    ]
                if precision in ("16", 16, "fp16"):
                    optimize_command.append("--compress_to_fp16")
                _ = run(args=optimize_command, check=False)
                bin_file = Path(ir_dir) / f"{onnx_key}.bin"
                xml_file = Path(ir_dir) / f"{onnx_key}.xml"
                if bin_file.exists() and xml_file.exists():
                    results["outputs"]["openvino"][onnx_key] = {}
                    results["outputs"]["openvino"][onnx_key]["bin"] = str(Path(ir_dir) / f"{onnx_key}.bin")
                    results["outputs"]["openvino"][onnx_key]["xml"] = str(Path(ir_dir) / f"{onnx_key}.xml")
                else:
                    msg = "OpenVINO Export failed."
                    raise RuntimeError(msg)
        return results

    @property
    def callbacks(self) -> list[Callback]:
        """Returns a list of callbacks to be used.

        Returns:
            list[Callback]: A list of callbacks.
        """
        callback_list = [
            TQDMProgressBar(),
        ]
        if self.training:
            callback_list.extend(
                [
                    ModelCheckpoint(dirpath=self.work_dir, filename="{epoch:02d}", **self.config.callback.checkpoint),
                    LearningRateMonitor(),
                    EarlyStopping(**self.config.callback.early_stopping),
                ],
            )
        return callback_list
