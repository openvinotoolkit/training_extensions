"""Visual Prompting Task."""

# Copyright (C) 2023 Intel Corporation
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

import torch
from anomalib.models import AnomalyModule, get_model
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from otx.algorithms.anomaly.adapters.anomalib.callbacks import ProgressCallback
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets import (
    OTXVisualPromptingDataModule,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.training_interface import ITrainingTask

from .inference import InferenceTask

logger = get_logger()


class TrainingTask(InferenceTask, ITrainingTask):
    """Training Task for Visual Prompting."""

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: TrainParameters,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        """Train the visual prompting model.

        Args:
            dataset (DatasetEntity): Input dataset.
            output_model (ModelEntity): Output model to save the model weights.
            train_parameters (TrainParameters): Training parameters
            seed (Optional[int]): Setting seed to a value other than 0
            deterministic (bool): Setting PytorchLightning trainer's deterministic flag.
        """
        logger.info("Training the model.")

        if seed:
            logger.info(f"Setting seed to {seed}")
            seed_everything(seed, workers=True)
        self.config.trainer.deterministic = deterministic

        logger.info("Training Configs '%s'", self.config)
        
        datamodule = OTXVisualPromptingDataModule(config=self.config, dataset=dataset)
        callbacks = [
            # LearningRateMonitor(logging_interval='step'),
            ProgressCallback(parameters=train_parameters),
            ModelCheckpoint(monitor="iou", mode="max"),
            # MinMaxNormalizationCallback(),
            # MetricsConfigurationCallback(
            #     task=config.dataset.task,
            #     image_metrics=config.metrics.image,
            #     pixel_metrics=config.metrics.get("pixel"),
            # ),
            # PostProcessingConfigurationCallback(
            #     normalization_method=NormalizationMethod.MIN_MAX,
            #     threshold_method=ThresholdMethod.ADAPTIVE,
            #     manual_image_threshold=config.metrics.threshold.manual_image,
            #     manual_pixel_threshold=config.metrics.threshold.manual_pixel,
            # ),
        ]

        self.trainer = Trainer(**self.config.trainer, logger=False, callbacks=callbacks)
        self.trainer.fit(model=self.model, datamodule=datamodule)
        logger.info("Evaluation with best checkpoint.")
        self.trainer.validate(model=self.model, datamodule=datamodule)

        self.save_model(output_model)

        logger.info("Training completed.")
