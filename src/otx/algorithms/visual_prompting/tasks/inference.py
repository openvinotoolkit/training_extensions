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
import tempfile
import time
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import openvino as ov
import torch
from omegaconf import DictConfig, ListConfig
from openvino.tools import mo
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from otx.algorithms.common.configs.training_base import TrainType
from otx.algorithms.common.utils import set_random_seed
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.callbacks import (
    InferenceCallback,
    ZeroShotInferenceCallback,
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
from otx.api.entities.train_parameters import TrainParameters
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload
from otx.utils.logger import get_logger

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
        self.hyper_parameters: VisualPromptingBaseConfig = self.task_environment.get_hyper_parameters()
        self.train_type = self.hyper_parameters.algo_backend.train_type  # type: ignore[attr-defined]

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

        self.trainer: Trainer
        self._model_ckpt: Optional[str] = None

        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    def set_seed(self):
        """Set seed and deterministic."""
        if self.seed is None:
            # If the seed is not present via task.train, it will be found in the recipe.
            self.seed = self.config.get("seed", 5)
        if not self.deterministic:
            # deterministic is the same.
            self.deterministic = self.config.get("deterministic", False)
        self.config["seed"] = self.seed
        self.config["deterministic"] = self.deterministic
        set_random_seed(self.seed, logger, self.deterministic)

    def get_config(self) -> Union[DictConfig, ListConfig]:
        """Get Visual Prompting Config from task environment.

        Returns:
            Union[DictConfig, ListConfig]: Visual Prompting config.
        """
        # set checkpoints
        model_checkpoint: Optional[str] = None
        resume_from_checkpoint: Optional[str] = None
        if self.mode == "train" and self.task_environment.model is not None:
            # when args.load_weights or args.resume_from is set
            checkpoint_path = str(self.task_environment.model.model_adapters.get("path", None))
            if self.task_environment.model.model_adapters.get("resume", False):
                resume_from_checkpoint = checkpoint_path
            else:
                model_checkpoint = checkpoint_path

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

        def get_model(config: DictConfig, train_type: TrainType, state_dict: Optional[OrderedDict] = None):
            if config.model.name == "SAM":
                if train_type == TrainType.Incremental:
                    from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models import (
                        SegmentAnything as VisualPrompter,
                    )
                elif train_type == TrainType.Zeroshot:
                    from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models import (  # type: ignore[assignment] # noqa: E501
                        ZeroShotSegmentAnything as VisualPrompter,
                    )

                model = VisualPrompter(config=config, state_dict=state_dict)
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
        elif otx_model.model_adapters.get("resume", False):
            # If resuming, pass this part to load checkpoint in Trainer
            logger.info(f"To resume {otx_model.model_adapters.get('path')}, the checkpoint will be loaded in Trainer.")

        else:
            # Load state_dict
            buffer = io.BytesIO(otx_model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))
            if model_data.get("state_dict", None) and model_data.get("pytorch-lightning_version", None):
                # Load state_dict from pytorch lightning checkpoint or weights.pth saved by visual prompting task
                # In pytorch lightning checkpoint, there are metas: epoch, global_step, pytorch-lightning_version,
                # state_dict, loops, callbacks, optimizer_states, lr_schedulers, hparams_name, hyper_parameters.
                # To confirm if it is from pytorch lightning, check if one or two of them is in model_data.
                state_dict = model_data["state_dict"]

            elif model_data.get("model", None) and model_data.get("config", None):
                # Load state_dict from checkpoint saved by otx other tasks
                if model_data["config"]["model"]["backbone"] != self.config["model"]["backbone"]:
                    logger.warning(
                        "Backbone of the model in the Task Environment is different from the one in the template. "
                        f"creating model with backbone={model_data['config']['model']['backbone']}"
                    )
                    self.config["model"]["backbone"] = model_data["config"]["model"]["backbone"]
                state_dict = model_data["model"]

            else:
                # Load state_dict from naive pytorch checkpoint
                state_dict = model_data

        try:
            model = get_model(config=self.config, train_type=self.train_type, state_dict=state_dict)
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
        self.model = self.load_model(otx_model=self.task_environment.model)
        datamodule = OTXVisualPromptingDataModule(
            config=self.config.dataset, dataset=dataset, train_type=self.train_type
        )

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
                dummy_inputs = {"images": torch.randn(1, 3, height, width, dtype=torch.float32)}
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
                    "image_embeddings": torch.zeros(1, embed_dim, *embed_size, dtype=torch.float32),
                    "point_coords": torch.randint(low=0, high=1024, size=(1, 2, 2), dtype=torch.float32),
                    "point_labels": torch.randint(low=0, high=4, size=(1, 2), dtype=torch.float32),
                    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float32),
                    "has_mask_input": torch.tensor([[1]], dtype=torch.float32),
                    "orig_size": torch.randint(low=256, high=2048, size=(1, 2), dtype=torch.int64),
                }
                output_names = ["upscaled_masks", "iou_predictions", "low_res_masks"]
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
                        opset_version=13,
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

        self.model = self.load_model(otx_model=self.task_environment.model)
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
                mo_args: Dict[str, Any] = {"input_model": path}
                if module == "visual_prompting_image_encoder":
                    mo_args.update(
                        {
                            "mean_values": list(self.config.dataset.normalize.mean),
                            "scale_values": list(self.config.dataset.normalize.std),
                        }
                    )
                if precision == ModelPrecision.FP16:
                    mo_args.update({"compress_to_fp16": True})

                ov_model = mo.convert_model(**mo_args)
                ov.save_model(ov_model, os.path.join(self.output_path, f"{module}.xml"))
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
        if not self._model_ckpt:
            logger.warn("model checkpoint is not set, return empty dictionary.")
            return {}
        return torch.load(self._model_ckpt, map_location="cpu")

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


class ZeroShotTask(InferenceTask):
    """Learn task for Zero-shot learning.

    **There are two ways to be decided:
    1. use it independently <-- temporarily current setting
    2. use it depending on template

    The objective of this task is to get reference features and export it with decoder modules.
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

        self.seed = seed
        self.deterministic = deterministic
        self.set_seed()
        self.config.trainer.deterministic = "warn" if deterministic else deterministic

        logger.info(f"Training Configs {self.config}")

        self.model = self.load_model(otx_model=self.task_environment.model)

        datamodule = OTXVisualPromptingDataModule(
            config=self.config.dataset, dataset=dataset, train_type=self.train_type
        )

        self.trainer = Trainer(
            logger=CSVLogger(save_dir=self.output_path, name=".", version=self.timestamp), **self.config.trainer
        )
        self.trainer.fit(model=self.model, datamodule=datamodule)

        # save resulting model
        self.save_model(output_model)

    def infer(self, dataset: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        """Perform inference on a dataset.

        Args:
            dataset (DatasetEntity): Dataset to infer.
            inference_parameters (InferenceParameters): Inference parameters.

        Returns:
            DatasetEntity: Output dataset with predictions.
        """
        logger.info("Performing inference on the validation set using the base torch model.")
        self.model = self.load_model(otx_model=self.task_environment.model)
        datamodule = OTXVisualPromptingDataModule(
            config=self.config.dataset, dataset=dataset, train_type=self.train_type
        )

        logger.info("Inference Configs '%s'", self.config)

        # Callbacks
        inference_callback = ZeroShotInferenceCallback(
            otx_dataset=dataset, label_schema=self.task_environment.label_schema
        )
        callbacks = [TQDMProgressBar(), inference_callback]

        self.trainer = Trainer(**self.config.trainer, logger=False, callbacks=callbacks)
        self.trainer.predict(model=self.model, datamodule=datamodule)

        return inference_callback.otx_dataset

    def save_model(self, output_model: ModelEntity) -> None:
        """Save the model after training is completed.

        Args:
            output_model (ModelEntity): Output model onto which the weights are saved.
        """
        logger.info("Saving the model weights and reference features.")

        model_info = self.model.state_dict()
        model_info.pop("reference_info.reference_feats")
        model_info.pop("reference_info.used_indices")

        buffer = io.BytesIO()
        torch.save(model_info, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))

        output_model.precision = self.precision
        output_model.optimization_methods = self.optimization_methods
