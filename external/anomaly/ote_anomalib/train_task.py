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

import torch
from anomalib.utils.callbacks import MinMaxNormalizationCallback
from ote_anomalib import AnomalyInferenceTask
from ote_anomalib.callbacks import ProgressCallback
from ote_anomalib.data import OTEAnomalyDataModule
from ote_anomalib.logging import get_logger
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.metrics import Performance, ScoreMetric
from ote_sdk.entities.model import ModelEntity, ModelPrecision
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.serialization.label_mapper import label_schema_to_bytes
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from pytorch_lightning import Trainer

logger = get_logger(__name__)


class AnomalyTrainingTask(AnomalyInferenceTask, ITrainingTask):
    """Base Anomaly Task."""

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: TrainParameters,
    ) -> None:
        """Train the anomaly classification model.

        Args:
            dataset (DatasetEntity): Input dataset.
            output_model (ModelEntity): Output model to save the model weights.
            train_parameters (TrainParameters): Training parameters
        """
        logger.info("Training the model.")

        config = self.get_config()
        logger.info("Training Configs '%s'", config)

        datamodule = OTEAnomalyDataModule(config=config, dataset=dataset, task_type=self.task_type)
        callbacks = [ProgressCallback(parameters=train_parameters), MinMaxNormalizationCallback()]

        self.trainer = Trainer(**config.trainer, logger=False, callbacks=callbacks)
        self.trainer.fit(model=self.model, datamodule=datamodule)

        self.save_model(output_model)

        logger.info("Training completed.")

    def save_model(self, output_model: ModelEntity) -> None:
        """Save the model after training is completed.

        Args:
            output_model (ModelEntity): Output model onto which the weights are saved.
        """
        logger.info("Saving the model weights.")
        config = self.get_config()
        model_info = {
            "model": self.model.state_dict(),
            "config": config,
            "VERSION": 1,
        }
        buffer = io.BytesIO()
        torch.save(model_info, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))
        self._set_metadata(output_model)

        f1_score = self.model.image_metrics.F1.compute().item()
        output_model.performance = Performance(score=ScoreMetric(name="F1 Score", value=f1_score))
        output_model.precision = [ModelPrecision.FP32]
