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

import torch
from anomalib.models import AnomalyModule, get_model
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from pytorch_lightning import Trainer, seed_everything

from otx.algorithms.anomaly.adapters.anomalib.callbacks import ProgressCallback
from otx.algorithms.anomaly.adapters.anomalib.data import OTXAnomalyDataModule
from otx.algorithms.anomaly.adapters.anomalib.logger import get_logger
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.training_interface import ITrainingTask

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
        self.trainer.fit(model=self.model, datamodule=datamodule)

        self.save_model(output_model)

        logger.info("Training completed.")

    def load_model(self, otx_model: Optional[ModelEntity]) -> AnomalyModule:
        """Create and Load Anomalib Module from OTE Model.

        This method checks if the task environment has a saved OTE Model,
        and creates one. If the OTE model already exists, it returns the
        the model with the saved weights.

        Args:
            otx_model (Optional[ModelEntity]): OTE Model from the
                task environment.

        Returns:
            AnomalyModule: Anomalib
                classification or segmentation model with/without weights.
        """
        model = get_model(config=self.config)
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
