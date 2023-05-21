"""Anomaly Classification Task."""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import io
import random
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from segment_anything import sam_model_registry
from torch import nn, optim
from torchvision.utils import save_image

from anomalib.models import AnomalyModule
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from otx.algorithms.anomaly.adapters.anomalib.callbacks import ProgressCallback
from otx.algorithms.anomaly.adapters.anomalib.logger import get_logger
from otx.algorithms.visual_prompting.adapters.sam.data import OTXVisualPromptingDataModule
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.training_interface import ITrainingTask

from .inference import InferenceTask

ALPHA = 0.8
GAMMA = 2

logger = get_logger(__name__)


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    """Calc_iou."""
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


class FocalLoss(nn.Module):
    """FocalLoss."""

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        """Forward."""
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets.float(), reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):
    """DiceLoss."""

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        """Forward."""
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


focal_loss = FocalLoss()
dice_loss = DiceLoss()


class VisualPromptingModel(pl.LightningModule):
    """VisualPromptingModel."""

    def __init__(
        self,
        backbone="vit_b",
        checkpoint="/home/cosmos/ws/otx-sam/sam_vit_b_01ec64.pth",
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
        freeze_mask_decoder=False,
    ):
        super().__init__()
        self.model = sam_model_registry[backbone](checkpoint=checkpoint)
        self.model.train()
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        """Training_step."""
        images = batch["image"]
        boxes = batch["boxes"]
        gt_masks = batch["mask"]
        _, _, H, W = images.shape
        image_embeddings = self.model.image_encoder(images)
        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, boxes):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
        loss_focal = torch.tensor(0.0, device=self.model.device)
        loss_dice = torch.tensor(0.0, device=self.model.device)
        loss_iou = torch.tensor(0.0, device=self.model.device)

        one_hot_masks = F.one_hot(gt_masks.long(), num_classes=4)
        one_hot_masks = one_hot_masks.permute(0, 3, 1, 2)[:, 1:, :, :].contiguous()

        for pred_mask, gt_mask, iou_prediction in zip(pred_masks, one_hot_masks, iou_predictions):
            batch_iou = calc_iou(pred_mask, gt_mask)
            loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
            loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
            loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction="sum") / num_masks

            if random.random() < 0.5:
                save_image(gt_mask.unsqueeze(0).float() * 255, "gt.jpg")
                save_image((pred_mask >= 0.5).unsqueeze(0).float() * 255, "pred.jpg")
            loss_total = 20.0 * loss_focal + loss_dice + loss_iou

        loss_total = 20.0 * loss_focal + loss_dice + loss_iou
        self.log("train_loss", loss_total)
        return loss_total

    def configure_optimizers(self):
        """Configure_optimizers."""
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        return optimizer


class TrainingTask(InferenceTask, ITrainingTask):
    """Base Anomaly Task."""

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: TrainParameters,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        """Train the anomaly classification model.

        Args:
            dataset (DatasetEntity): Input dataset.
            output_model (ModelEntity): Output model to save the model weights.
            train_parameters (TrainParameters): Training parameters
            seed (Optional[int]): Setting seed to a value other than 0
            deterministic (bool): Setting PytorchLightning trainer's deterministic flag.
        """
        logger.info("Training the model.")

        config = self.get_config()

        if seed:
            logger.info(f"Setting seed to {seed}")
            seed_everything(seed, workers=True)

        config.trainer.deterministic = deterministic
        logger.info("Training Configs '%s'", config)

        datamodule = OTXVisualPromptingDataModule(config=config, dataset=dataset, task_type=self.task_type)
        callbacks = [
            ProgressCallback(parameters=train_parameters),
            MinMaxNormalizationCallback(),
            MetricsConfigurationCallback(
                task=config.dataset.task,
                image_metrics=config.metrics.image,
                pixel_metrics=config.metrics.get("pixel"),
            ),
            PostProcessingConfigurationCallback(
                normalization_method=NormalizationMethod.MIN_MAX,
                threshold_method=ThresholdMethod.ADAPTIVE,
                manual_image_threshold=config.metrics.threshold.manual_image,
                manual_pixel_threshold=config.metrics.threshold.manual_pixel,
            ),
        ]
        self.trainer = Trainer(**config.trainer, logger=False, callbacks=callbacks)
        self.trainer.fit(model=VisualPromptingModel(), datamodule=datamodule)

        self.save_model(output_model)

        logger.info("Training completed.")

    def load_model(self, otx_model: Optional[ModelEntity]) -> AnomalyModule:
        """Create and Load Anomalib Module from OTX Model.

        This method checks if the task environment has a saved OTX Model,
        and creates one. If the OTX model already exists, it returns the
        the model with the saved weights.

        Args:
            otx_model (Optional[ModelEntity]): OTX Model from the
                task environment.

        Returns:
            AnomalyModule: Anomalib
                classification or segmentation model with/without weights.
        """
        # breakpoint()
        # model = get_model(config=self.config)
        model = VisualPromptingModel()
        if otx_model is None:
            logger.info(
                "No trained model in project yet. Created new model with '%s'",
                self.model_name,
            )
        else:
            buffer = io.BytesIO(otx_model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))

            try:
                if model_data["config"]["model"]["backbone"] == self.config["model"]["backbone"]:
                    model.load_state_dict(model_data["model"])
                    logger.info("Loaded model weights from Task Environment")
                else:
                    logger.info(
                        "Model backbone does not match. Created new model with '%s'",
                        self.model_name,
                    )
            except BaseException as exception:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") from exception

        return model
