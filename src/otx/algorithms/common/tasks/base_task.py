"""Base task of OTX."""

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
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import distributed as dist

from otx.algorithms.common.adapters.mmcv.hooks import OTXLoggerHook
from otx.algorithms.common.adapters.mmcv.hooks.cancel_hook import CancelInterfaceHook
from otx.algorithms.common.configs.training_base import TrainType
from otx.algorithms.common.utils import UncopiableDefaultDict, append_dist_rank_suffix, set_random_seed
from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.explain_parameters import ExplainParameters
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import LabelEntity
from otx.api.entities.metrics import MetricsGroup
from otx.api.entities.model import ModelEntity, ModelFormat, ModelOptimizationType, ModelPrecision, OptimizationMethod
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.train_parameters import TrainParameters
from otx.api.serialization.label_mapper import LabelSchemaMapper
from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload

TRAIN_TYPE_DIR_PATH = {
    TrainType.Incremental.name: ".",
    TrainType.Selfsupervised.name: "selfsl",
    TrainType.Semisupervised.name: "semisl",
}

logger = get_logger()


class OnHookInitialized:
    """OnHookInitialized class."""

    def __init__(self, task_instance):
        self.task_instance = task_instance
        self.__findable = False  # a barrier to block segmentation fault

    def __call__(self, cancel_interface):
        """Function call in OnHookInitialized."""
        if isinstance(self.task_instance, int) and self.__findable:
            import ctypes

            # NOTE: BE AWARE OF SEGMENTATION FAULT
            self.task_instance = ctypes.cast(self.task_instance, ctypes.py_object).value
        self.task_instance.cancel_hook_initialized(cancel_interface)

    def __repr__(self):
        """Function repr in OnHookInitialized."""
        return f"'{__name__}.OnHookInitialized'"

    def __deepcopy__(self, memo):
        """Function deepcopy in OnHookInitialized."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.task_instance = self.task_instance
        result.__findable = True  # pylint: disable=unused-private-member, protected-access
        return result

    def __reduce__(self):
        """Function reduce in OnHookInitialized."""
        return (self.__class__, (id(self.task_instance),))


# pylint: disable=too-many-instance-attributes
class OTXTask(IInferenceTask, IExportTask, IEvaluationTask, IUnload, ABC):
    """Base task of OTX."""

    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        self._config: Dict[Any, Any] = {}
        self._task_environment = task_environment
        self._task_type = task_environment.model_template.task_type
        self._labels = task_environment.get_labels(include_empty=False)
        self._work_dir_is_temp = False
        self._output_path = output_path
        self._output_path = output_path if output_path is not None else self._get_tmp_dir()
        self._time_monitor: Optional[TimeMonitorCallback] = None
        self.on_hook_initialized = OnHookInitialized(self)
        self._learning_curves = UncopiableDefaultDict(OTXLoggerHook.Curve)
        self._model_label_schema: List[LabelEntity] = []
        self._resume = False
        self._should_stop = False
        self.cancel_interface: Optional[CancelInterfaceHook] = None
        self.reserved_cancel = False
        self._model_ckpt = None
        self._precision = [ModelPrecision.FP32]
        self._optimization_methods: List[OptimizationMethod] = []
        self._is_training = False
        self.seed: Optional[int] = None
        self.deterministic: bool = False

        self.override_configs: Dict[str, str] = {}

        # This is for hpo, and this should be removed
        self.project_path = self._output_path

        if self._is_distributed_training():
            self._setup_distributed_training()

    @staticmethod
    def _is_distributed_training():
        multi_gpu_env = ["MASTER_ADDR", "MASTER_PORT", "LOCAL_WORLD_SIZE", "WORLD_SIZE", "LOCAL_RANK", "RANK"]
        for env in multi_gpu_env:
            if env not in os.environ:
                return False

        return torch.cuda.is_available()

    @staticmethod
    def _setup_distributed_training():
        if not dist.is_initialized():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=30))
            rank = dist.get_rank()
            logger.info(f"Dist info: rank {rank} / {dist.get_world_size()} world_size")
            if rank != 0:
                logging.disable(logging.WARNING)

    def _get_tmp_dir(self):
        self._work_dir_is_temp = True
        # If training is excuted with torchrun, set all trainings' output directory same
        if "TORCHELASTIC_RUN_ID" in os.environ:
            return os.path.join(tempfile.gettempdir(), f"OTX-task-torchelastic-{os.environ['TORCHELASTIC_RUN_ID']}")
        return tempfile.mkdtemp(prefix="OTX-task-")

    def _load_model(self):
        """Loading model from checkpoint."""

        def _load_model_label_schema(model: Optional[ModelEntity]):
            # If a model has been trained and saved for the task already, create empty model and load weights here
            if model and "label_schema.json" in model.model_adapters:
                import json

                buffer = json.loads(model.get_data("label_schema.json").decode("utf-8"))
                model_label_schema = LabelSchemaMapper().backward(buffer)
                return model_label_schema.get_labels(include_empty=False)
            return self._labels

        logger.info("loading the model from the task env.")
        model = self._task_environment.model
        state_dict = self._load_model_ckpt(model)
        if state_dict:
            self._model_ckpt = append_dist_rank_suffix(os.path.join(self._output_path, "env_model_ckpt.pth"))
            if os.path.exists(self._model_ckpt):
                os.remove(self._model_ckpt)
            torch.save(state_dict, self._model_ckpt)
            self._model_label_schema = _load_model_label_schema(model)
            if model is not None:
                self._resume = model.model_adapters.get("resume", False)

    def _load_model_ckpt(self, model: Optional[ModelEntity]):
        if model and "weights.pth" in model.model_adapters:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))
            return model_data
        return None

    @abstractmethod
    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: Optional[TrainParameters] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ):
        """Train function for OTX task."""
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Main infer function."""
        raise NotImplementedError

    @abstractmethod
    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = True,
    ):
        """Export function of OTX Task."""
        raise NotImplementedError

    @abstractmethod
    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[ExplainParameters] = None,
    ) -> DatasetEntity:
        """Main explain function of OTX Task."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        output_resultset: ResultSetEntity,
        evaluation_metric: Optional[str] = None,
    ):
        """Evaluate function of OTX Task."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _generate_training_metrics(learning_curves, scores) -> Iterable[MetricsGroup[Any, Any]]:
        """Get Training metrics (epochs & scores).

        Parses the training logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, output_model: ModelEntity):
        """Save best model weights in trining task."""
        raise NotImplementedError

    def cancel_training(self):
        """Cancel training function in trining task.

        Sends a cancel training signal to gracefully stop the optimizer. The signal consists of creating a
        '.stop_training' file in the current work_dir. The runner checks for this file periodically.
        The stopping mechanism allows stopping after each iteration, but validation will still be carried out. Stopping
        will therefore take some time.
        """
        logger.info("Cancel training requested.")
        self._should_stop = True
        if self.cancel_interface is not None:
            self.cancel_interface.cancel()
        else:
            logger.info("but training was not started yet. reserved it to cancel")
            self.reserved_cancel = True

    def cancel_hook_initialized(self, cancel_interface: CancelInterfaceHook):
        """Initialization of cancel_interface hook."""
        logger.info("cancel hook is initialized")
        self.cancel_interface = cancel_interface
        if self.reserved_cancel and self.cancel_interface:
            self.cancel_interface.cancel()

    def cleanup(self):
        """Clean up work directory if user specified it."""
        if self._work_dir_is_temp:
            self._delete_scratch_space()

    def _delete_scratch_space(self):
        """Remove model checkpoints and otx logs."""
        if os.path.exists(self._output_path):
            shutil.rmtree(self._output_path, ignore_errors=False)

    def unload(self):
        """Unload the task."""
        self.cleanup()
        if self._is_docker():
            logger.warning("Got unload request. Unloading models. Throwing Segmentation Fault on purpose")
            import ctypes

            ctypes.string_at(0)
        else:
            logger.warning("Got unload request, but not on Docker. Only clearing CUDA cache")
            torch.cuda.empty_cache()
            logger.warning(
                f"Done unloading. " f"Torch is still occupying {torch.cuda.memory_allocated()} bytes of GPU memory"
            )

    @staticmethod
    def _is_docker():
        """Checks whether the task runs in docker container.

        :return bool: True if task runs in docker
        """
        path = "/proc/self/cgroup"
        is_in_docker = False
        if os.path.isfile(path):
            with open(path, encoding="UTF-8") as f:
                is_in_docker = is_in_docker or any("docker" in line for line in f)
        is_in_docker = is_in_docker or os.path.exists("/.dockerenv")
        return is_in_docker

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

    @property
    def config(self):
        """Config of OTX task."""
        return self._config

    @config.setter
    def config(self, config: Dict[Any, Any]):
        self._config = config

    def _update_model_export_metadata(
        self, output_model: ModelEntity, export_type: ExportType, precision: ModelPrecision, dump_features: bool
    ) -> None:
        """Updates a model entity with format and optimization related attributes."""
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

        output_model.has_xai = dump_features
        output_model.optimization_methods = self._optimization_methods
        output_model.precision = [precision]
