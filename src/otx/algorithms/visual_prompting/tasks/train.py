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

from typing import Optional

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets import (
    OTXVisualPromptingDataModule,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.metrics import Performance, ScoreMetric
from otx.api.entities.model import ModelEntity
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.training_interface import ITrainingTask

from .inference import InferenceTask

logger = get_logger()


class TrainingTask(InferenceTask, ITrainingTask):
    """Training Task for Visual Prompting.

    Args:
        dataset (DatasetEntity): Input dataset.
        output_model (ModelEntity): Output model to save the model weights.
        train_parameters (TrainParameters): Training parameters
        seed (Optional[int]): Setting seed to a value other than 0
        deterministic (bool): Setting PytorchLightning trainer's deterministic flag.
    """

    def train(  # noqa: D102
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: TrainParameters,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:

        logger.info("Training the model.")

        if seed:
            logger.info(f"Setting seed to {seed}")
            seed_everything(seed, workers=True)
        self.config.trainer.deterministic = deterministic

        logger.info("Training Configs '%s'", self.config)

        datamodule = OTXVisualPromptingDataModule(config=self.config.dataset, dataset=dataset)
        loggers = CSVLogger(save_dir=self.output_path, name=".", version=self.timestamp)
        callbacks = [
            TQDMProgressBar(),
            ModelCheckpoint(dirpath=loggers.log_dir, filename="{epoch:02d}", **self.config.callback.checkpoint),
            LearningRateMonitor(),
            EarlyStopping(**self.config.callback.early_stopping),
        ]

        self.trainer = Trainer(**self.config.trainer, logger=loggers, callbacks=callbacks)
        self.trainer.fit(model=self.model, datamodule=datamodule)

        model_ckpt = self.trainer.checkpoint_callback.best_model_path
        if not model_ckpt:
            logger.error("cannot find final checkpoint from the results.")
            return
        # update checkpoint to the newly trained model
        self._model_ckpt = model_ckpt

        # compose performance statistics
        best_score = self.trainer.checkpoint_callback.best_model_score
        if best_score is None:
            results = self.trainer.validate(model=self.model, datamodule=datamodule)
            best_score = results[0].get(self.config.callback.checkpoint.monitor)

        # save resulting model
        self.save_model(output_model)
        performance = Performance(
            score=ScoreMetric(value=best_score, name=self.trainer.checkpoint_callback.monitor)
            # TODO (sungchul): dashboard? -> only for Geti
        )
        logger.info(f"Final model performance: {str(performance)}")
        output_model.performance = performance

        logger.info("train done.")
