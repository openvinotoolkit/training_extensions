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

from typing import Optional

from otx.algorithms.anomaly.adapters.anomalib.callbacks import ProgressCallback, ScoreReportingCallback
from otx.algorithms.anomaly.adapters.anomalib.data import OTXAnomalyDataModule
from otx.algorithms.anomaly.adapters.anomalib.logger import get_logger
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.training_interface import ITrainingTask
from pytorch_lightning import Trainer, seed_everything

from .inference import InferenceTask

logger = get_logger(__name__)


class TrainingTask(InferenceTask, ITrainingTask):
    """Base Anomaly Task."""

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: TrainParameters,
        seed: Optional[int] = None,
    ) -> None:
        """Train the anomaly classification model.

        Args:
            dataset (DatasetEntity): Input dataset.
            output_model (ModelEntity): Output model to save the model weights.
            train_parameters (TrainParameters): Training parameters
            seed: (Optional[int]): Setting seed to a value other than 0 also marks PytorchLightning trainer's
                deterministic flag to True.
        """
        logger.info("Training the model.")

        config = self.get_config()

        if seed:
            logger.info(f"Setting seed to {seed}")
            seed_everything(seed, workers=True)
            config.trainer.deterministic = True

        logger.info("Training Configs '%s'", config)

        datamodule = OTXAnomalyDataModule(config=config, dataset=dataset, task_type=self.task_type)
        callbacks = [
            ProgressCallback(parameters=train_parameters),
            MinMaxNormalizationCallback(),
            ScoreReportingCallback(parameters=train_parameters),
            MetricsConfigurationCallback(
                adaptive_threshold=config.metrics.threshold.adaptive,
                default_image_threshold=config.metrics.threshold.image_default,
                default_pixel_threshold=config.metrics.threshold.pixel_default,
                image_metric_names=config.metrics.image,
                pixel_metric_names=config.metrics.pixel,
            ),
        ]

        self.trainer = Trainer(**config.trainer, logger=False, callbacks=callbacks)
        self.trainer.fit(model=self.model, datamodule=datamodule)

        self.save_model(output_model)

        logger.info("Training completed.")
