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

from anomalib.utils.callbacks import MinMaxNormalizationCallback
from ote_anomalib import AnomalyInferenceTask
from ote_anomalib.callbacks import ProgressCallback, ScoreReportingCallback
from ote_anomalib.data import OTEAnomalyDataModule
from ote_anomalib.logging import get_logger
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from pytorch_lightning import Trainer, seed_everything

logger = get_logger(__name__)


class AnomalyTrainingTask(AnomalyInferenceTask, ITrainingTask):
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

        datamodule = OTEAnomalyDataModule(config=config, dataset=dataset, task_type=self.task_type)
        callbacks = [
            ProgressCallback(parameters=train_parameters),
            MinMaxNormalizationCallback(),
            ScoreReportingCallback(parameters=train_parameters)
        ]

        self.trainer = Trainer(**config.trainer, logger=False, callbacks=callbacks)
        self.trainer.fit(model=self.model, datamodule=datamodule)

        self.save_model(output_model)

        logger.info("Training completed.")
