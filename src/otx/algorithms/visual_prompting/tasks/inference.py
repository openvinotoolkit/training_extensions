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

import ctypes
import io
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.callbacks import (
    InferenceCallback,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.config import (
    get_visual_promtping_config,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets import (
    OTXVisualPromptingDataModule,
)
from otx.algorithms.visual_prompting.configs.base.configuration import (
    VisualPromptingBaseConfig,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import (
    ModelEntity,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload

logger = get_logger()


# pylint: disable=too-many-instance-attributes
class InferenceTask(IInferenceTask, IEvaluationTask, IExportTask, IUnload):
    """Base Visual Prompting Task.

    Train, Infer, Export, Optimize and Deploy an Visual Prompting Task.

    Args:
        task_environment (TaskEnvironment): OTX Task environment.
        output_path (Optional[str]): output path where task output are saved.
    """

    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None) -> None:
        torch.backends.cudnn.enabled = True
        logger.info("Initializing the task environment.")
        self.task_environment = task_environment
        self.task_type = task_environment.model_template.task_type
        self.model_name = task_environment.model_template.name
        self.labels = task_environment.get_labels()

        template_file_path = task_environment.model_template.model_template_path
        self.base_dir = os.path.abspath(os.path.dirname(template_file_path))

        # Hyperparameters.
        self._work_dir_is_temp = False
        self.output_path = output_path
        if self.output_path is None:
            self.output_path = tempfile.mkdtemp(prefix="otx-visual_prompting")
            self._work_dir_is_temp = True
        self.config = self.get_config()

        # Set default model attributes.
        self.optimization_methods: List[OptimizationMethod] = []
        self.precision = [ModelPrecision.FP32]
        self.optimization_type = ModelOptimizationType.MO

        self.model = self.load_model(otx_model=task_environment.model)

        self.trainer: Trainer

        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    def get_config(self) -> Union[DictConfig, ListConfig]:
        """Get Visual Prompting Config from task environment.

        Returns:
            Union[DictConfig, ListConfig]: Visual Prompting config.
        """
        self.hyper_parameters: VisualPromptingBaseConfig = self.task_environment.get_hyper_parameters()

        # set checkpoints
        resume_from_checkpoint = model_checkpoint = None
        if self.task_environment.model is not None:
            # self.task_environment.model is set for two cases:
            # 1. otx train : args.load_weights or args.resume_from
            #   - path and resume are set into model_adapters
            # 2. otx eval
            #   - both resume_from and path are not needed to be set, path is not set into model_adapters,
            #     so it can be distinguished by using it
            resume_from_checkpoint = model_checkpoint = self.task_environment.model.model_adapters.get("path", None)
            if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.endswith(".pth"):
                # TODO (sungchul): support resume from checkpoint
                logger.info("[*] Pytorch checkpoint cannot be used for resuming. It will be supported.")
                resume_from_checkpoint = None
            elif not self.task_environment.model.model_adapters.get("resume", False):
                # If not resuming, set resume_from_checkpoint to None to avoid training in resume environment
                # and saving to configuration.
                resume_from_checkpoint = None
            else:
                # If resuming, set model_checkpoint to None to avoid loading weights twice and saving to configuration.
                model_checkpoint = None

        config = get_visual_promtping_config(
            self.model_name,
            self.hyper_parameters,
            self.output_path,  # type: ignore[arg-type]
            model_checkpoint,  # type: ignore[arg-type]
            resume_from_checkpoint,  # type: ignore[arg-type]
        )

        config.dataset.task = "visual_prompting"

        return config

    def load_model(self, otx_model: Optional[ModelEntity] = None) -> LightningModule:
        """Create and Load Visual Prompting Module.

        Currently, load model through `sam_model_registry` because there is only SAM.
        If other visual prompting model is added, loading model process must be changed.

        Args:
            otx_model (Optional[ModelEntity]): OTX Model from the task environment.

        Returns:
            LightningModule: Visual prompting model with/without weights.
        """

        def get_model(config: DictConfig, state_dict: Optional[OrderedDict] = None):
            if config.model.name == "SAM":
                from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models import (
                    SegmentAnything,
                )

                model = SegmentAnything(config=config, state_dict=state_dict)
            else:
                raise NotImplementedError(
                    (f"Current selected model {config.model.name} is not implemented. " f"Use SAM instead.")
                )
            return model

        state_dict = None
        if otx_model is None:
            logger.info(
                "No trained model in project yet. Created new model with '%s'",
                self.model_name,
            )
        elif ("path" in otx_model.model_adapters) and (
            otx_model.model_adapters.get("path").endswith(".ckpt")  # type: ignore[attr-defined]
        ):
            # pytorch lightning checkpoint
            if not otx_model.model_adapters.get("resume"):
                # If not resuming, just load weights in LightningModule
                logger.info("Load pytorch lightning checkpoint.")
        else:
            # pytorch checkpoint saved by otx
            buffer = io.BytesIO(otx_model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))
            if model_data.get("model", None) and model_data.get("config", None):
                if model_data["config"]["model"]["backbone"] != self.config["model"]["backbone"]:
                    logger.warning(
                        "Backbone of the model in the Task Environment is different from the one in the template. "
                        f"creating model with backbone={model_data['config']['model']['backbone']}"
                    )
                    self.config["model"]["backbone"] = model_data["config"]["model"]["backbone"]
                state_dict = model_data["model"]
                logger.info("Load pytorch checkpoint from weights.pth.")
            else:
                state_dict = model_data
                logger.info("Load pytorch checkpoint.")

        try:
            model = get_model(config=self.config, state_dict=state_dict)
            logger.info("Complete to load model.")
        except BaseException as exception:
            raise ValueError("Could not load the saved model. The model file structure is invalid.") from exception

        return model

    def cancel_training(self) -> None:  # noqa: D102
        raise NotImplementedError

    def infer(self, dataset: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        """Perform inference on a dataset.

        Args:
            dataset (DatasetEntity): Dataset to infer.
            inference_parameters (InferenceParameters): Inference parameters.

        Returns:
            DatasetEntity: Output dataset with predictions.
        """
        logger.info("Performing inference on the validation set using the base torch model.")
        datamodule = OTXVisualPromptingDataModule(config=self.config.dataset, dataset=dataset)

        logger.info("Inference Configs '%s'", self.config)

        # Callbacks
        inference_callback = InferenceCallback(otx_dataset=dataset)
        callbacks = [TQDMProgressBar(), inference_callback]

        self.trainer = Trainer(**self.config.trainer, logger=False, callbacks=callbacks)
        self.trainer.predict(model=self.model, datamodule=datamodule)

        return inference_callback.otx_dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None) -> None:
        """Evaluate the performance on a result set.

        Args:
            output_resultset (ResultSetEntity): Result Set from which the performance is evaluated.
            evaluation_metric (Optional[str], optional): Evaluation metric. Defaults to None. Instead,
                metric is chosen depending on the task type.
        """
        metric = MetricsHelper.compute_dice_averaged_over_pixels(output_resultset)
        logger.info(f"mDice after evaluation: {metric.overall_dice.value}")
        output_resultset.performance = metric.get_performance()
        logger.info("Evaluation completed")

    def _export_to_onnx(self, onnx_path: str):
        raise NotImplementedError

    def export(  # noqa: D102
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = False,
    ) -> None:
        raise NotImplementedError

    def model_info(self) -> Dict:
        """Return model info to save the model weights.

        Returns:
           Dict: Model info.
        """
        return {
            "model": self.model.state_dict(),
            "config": self.get_config(),
            "version": self.trainer.logger.version,
        }

    def save_model(self, output_model: ModelEntity) -> None:
        """Save the model after training is completed.

        Args:
            output_model (ModelEntity): Output model onto which the weights are saved.
        """
        logger.info("Saving the model weights.")
        model_info = self.model_info()
        buffer = io.BytesIO()
        torch.save(model_info, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))

        output_model.precision = self.precision
        output_model.optimization_methods = self.optimization_methods

    @staticmethod
    def _is_docker() -> bool:
        raise NotImplementedError

    def unload(self) -> None:
        """Unload the task."""
        self.cleanup()

        if self._is_docker():
            logger.warning("Got unload request. Unloading models. Throwing Segmentation Fault on purpose")
            ctypes.string_at(0)

        else:
            logger.warning("Got unload request, but not on Docker. Only clearing CUDA cache")
            torch.cuda.empty_cache()
            logger.warning(
                "Done unloading. Torch is still occupying %f bytes of GPU memory",
                torch.cuda.memory_allocated(),
            )

    def cleanup(self) -> None:
        """Clean up work directory."""
        if self._work_dir_is_temp:
            self._delete_scratch_space()

    def _delete_scratch_space(self) -> None:
        """Remove model checkpoints and otx logs."""
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path, ignore_errors=False)
