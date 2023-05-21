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
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from segment_anything import SamPredictor, sam_model_registry
from torch import optim

from anomalib.models import AnomalyModule
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from otx.algorithms.anomaly.adapters.anomalib.callbacks import ProgressCallback
from otx.algorithms.anomaly.adapters.anomalib.logger import get_logger

# from otx.algorithms.anomaly.adapters.anomalib.data import OTXAnomalyDataModule
from otx.algorithms.visual_prompting.adapters.sam.data import OTXVisualPromptingDataModule
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.training_interface import ITrainingTask

from .inference import InferenceTask

logger = get_logger(__name__)


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
        pass

    def configure_optimizers(self):
        """Configure_optimizers."""
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def forward(self, images, bboxes):
        """Forward."""
        _, _, H, W = images.shape
        image_embeddings = self.model.image_encoder(images)
        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, bboxes):
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

        return pred_masks, ious

    def get_predictor(self):
        """Get predictor."""
        return SamPredictor(self.model)


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
