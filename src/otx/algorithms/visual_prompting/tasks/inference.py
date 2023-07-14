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
import json
import os
import shutil
import subprocess
import tempfile
import time
import warnings
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
    ModelFormat,
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

    Train, Infer, and Export an Visual Prompting Task.

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
        self.mode = "train"
        if task_environment.model is not None and task_environment.model.train_dataset is None:
            self.mode = "export"
        if self.output_path is None:
            self.output_path = tempfile.mkdtemp(prefix="otx-visual_prompting")
            self._work_dir_is_temp = True
            self.mode = "inference"
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
        model_checkpoint: Optional[str] = None
        resume_from_checkpoint: Optional[str] = None
        if self.mode == "train" and self.task_environment.model is not None:
            # when args.load_weights or args.resume_from is set
            resume_from_checkpoint = model_checkpoint = self.task_environment.model.model_adapters.get("path", None)  # type: ignore  # noqa: E501
            if self.task_environment.model.model_adapters.get("resume", False):
                if resume_from_checkpoint.endswith(".pth"):  # type: ignore
                    logger.info("[*] Pytorch checkpoint cannot be used for resuming. It will be supported.")
                    resume_from_checkpoint = None
                else:
                    model_checkpoint = None
            else:
                # If not resuming, set resume_from_checkpoint to None to avoid training in resume environment
                # and saving to configuration.
                resume_from_checkpoint = None

        config = get_visual_promtping_config(
            task_name=self.model_name,
            otx_config=self.hyper_parameters,
            config_dir=self.base_dir,
            mode=self.mode,
            model_checkpoint=model_checkpoint,
            resume_from_checkpoint=resume_from_checkpoint,
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

    def _export_to_onnx(self, onnx_path: Dict[str, str]):
        """Export model to ONNX.

        Args:
             onnx_path (Dict[str, str]): Paths to save ONNX models.
        """
        height = width = self.config.model.image_size
        for module, path in onnx_path.items():
            if module == "visual_prompting_image_encoder":
                dummy_inputs = {"images": torch.randn(1, 3, height, width, dtype=torch.float)}
                output_names = ["image_embeddings"]
                dynamic_axes = None
                model_to_export = self.model.image_encoder

            else:
                # sam without backbone
                embed_dim = self.model.prompt_encoder.embed_dim
                embed_size = self.model.prompt_encoder.image_embedding_size
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
                output_names = ["iou_predictions", "low_res_masks"]
                model_to_export = self.model

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                with open(path, "wb") as f:
                    torch.onnx.export(
                        model_to_export,
                        tuple(dummy_inputs.values()),
                        f,
                        export_params=True,
                        verbose=False,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                    )

    def export(  # noqa: D102
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = False,
    ) -> None:
        """Export model to OpenVINO IR.

        When SAM gets an image for inference, image encoder runs just once to get image embedding.
        After that, prompt encoder + mask decoder runs repeatedly to get mask prediction.
        For this case, SAM should be divided into two parts, image encoder and prompt encoder + mask decoder.

        Args:
            export_type (ExportType): Export type should be ExportType.OPENVINO
            output_model (ModelEntity): The model entity in which to write the OpenVINO IR data
            precision (bool): Output model weights and inference precision
            dump_features (bool): Flag to return "feature_vector" and "saliency_map".

        Raises:
            Exception: If export_type is not ExportType.OPENVINO
        """
        if dump_features:
            logger.warning(
                "Feature dumping is not implemented for the visual prompting task."
                "The saliency maps and representation vector outputs will not be dumped in the exported model."
            )

        if export_type == ExportType.ONNX:
            output_model.model_format = ModelFormat.ONNX
            output_model.optimization_type = ModelOptimizationType.ONNX
            if precision == ModelPrecision.FP16:
                raise RuntimeError("Export to FP16 ONNX is not supported")
        elif export_type == ExportType.OPENVINO:
            output_model.model_format = ModelFormat.OPENVINO
            output_model.optimization_type = ModelOptimizationType.MO
        else:
            raise RuntimeError(f"not supported export type {export_type}")

        self.precision[0] = precision
        output_model.has_xai = dump_features

        logger.info("Exporting to the OpenVINO model.")
        onnx_path = {
            "visual_prompting_image_encoder": os.path.join(self.output_path, "visual_prompting_image_encoder.onnx"),
            "visual_prompting_decoder": os.path.join(self.output_path, "visual_prompting_decoder.onnx"),
        }
        self._export_to_onnx(onnx_path)

        if export_type == ExportType.ONNX:
            for module, path in onnx_path.items():
                with open(path, "rb") as file:
                    output_model.set_data(f"{module}.onnx", file.read())
        else:
            for module, path in onnx_path.items():
                optimize_command = [
                    "mo",
                    "--input_model",
                    path,
                    "--output_dir",
                    self.output_path,
                    "--model_name",
                    module,
                ]
                if module == "visual_prompting_image_encoder":
                    optimize_command += [
                        "--mean_values",
                        str(self.config.dataset.normalize.mean).replace(", ", ","),
                        "--scale_values",
                        str(self.config.dataset.normalize.std).replace(", ", ","),
                    ]
                if precision == ModelPrecision.FP16:
                    optimize_command.append("--compress_to_fp16")
                subprocess.run(optimize_command, check=True)
                with open(path.replace(".onnx", ".bin"), "rb") as file:
                    output_model.set_data(f"{module}.bin", file.read())
                with open(path.replace(".onnx", ".xml"), "rb") as file:
                    output_model.set_data(f"{module}.xml", file.read())

        output_model.precision = self.precision
        output_model.optimization_methods = self.optimization_methods

        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))
        self._set_metadata(output_model)

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

    def _set_metadata(self, output_model: ModelEntity) -> None:
        """Set metadata to the output model."""
        metadata = {"image_size": int(self.config.dataset.image_size)}

        # Set the task type for inferencer
        metadata["task"] = str(self.task_type).lower().split("_")[-1]  # type: ignore
        output_model.set_data("metadata", json.dumps(metadata).encode())

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
