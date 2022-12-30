"""NNCF Task of OTX Segmentation."""

# Copyright (C) 2022 Intel Corporation
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

import copy
import io
import json
import logging
import os
import tempfile
from typing import DefaultDict, List, Optional

import torch
from mmcv.runner import load_checkpoint, load_state_dict
from mmcv.utils import Config, ConfigDict
from mmseg.apis import train_segmentor
from mmseg.apis.train import build_val_dataloader
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.integration.nncf import (
    check_nncf_is_enabled,
    is_accuracy_aware_training_set,
    is_state_nncf,
    wrap_nncf_model,
)
from mmseg.integration.nncf.config import compose_nncf_config
from mmseg.models import build_segmentor

from otx.algorithms.common.adapters.mmcv.hooks import OTXLoggerHook
from otx.algorithms.common.utils.callback import OptimizationProgressCallback
from otx.algorithms.segmentation.adapters.mmseg.utils.config_utils import (
    patch_config,
    prepare_for_training,
    set_hyperparams,
)
from otx.algorithms.segmentation.configs.base import SegmentationConfig
from otx.algorithms.segmentation.tasks import SegmentationInferenceTask
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import ids_to_strings
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.optimization_parameters import default_progress_callback
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationParameters,
    OptimizationType,
)
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from otx.mpa.utils.config_utils import remove_custom_hook

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes, too-many-ancestors
class SegmentationNNCFTask(SegmentationInferenceTask, IOptimizationTask):
    """Task for compressing object detection models using NNCF."""

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment, **kwargs):
        super().__init__(task_environment, **kwargs)

        self._val_dataloader = None
        self._compression_ctrl = None
        self._nncf_preset = "nncf_quantization"
        check_nncf_is_enabled()
        # super().__init__(task_environment)
        self._scratch_space = tempfile.mkdtemp(prefix="otx-seg-scratch-")
        logger.info(f"Scratch space created at {self._scratch_space}")

        self._task_environment = task_environment

        self._model_name = task_environment.model_template.name
        self._labels = task_environment.get_labels(include_empty=False)
        self._label_dictionary = dict(enumerate(self._labels, 1))

        template_file_path = task_environment.model_template.model_template_path

        # Get and prepare mmseg config.
        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))
        config_file_path = os.path.join(self._base_dir, "model.py")
        self._config = Config.fromfile(config_file_path)

        # Align MPA config for nncf task
        self._initialize()
        self._config = Config()
        if self._recipe_cfg:
            self._config = self._recipe_cfg
            self._config.merge_from_dict(self._model_cfg)
            remove_custom_hook(self._config, "CancelInterfaceHook")
        else:
            config_file_path = os.path.join(self._base_dir, "model.py")
            self._config = Config.fromfile(config_file_path)
        # Disable task adaptation in NNCF task
        if hasattr(self._config.model, "is_task_adapt"):
            self._config.model.is_task_adapt = False

        distributed = torch.distributed.is_initialized()
        patch_config(self._config, self._scratch_space, self._labels, random_seed=42, distributed=distributed)
        set_hyperparams(self._config, self._hyperparams)

        # Set default model attributes.
        self._optimization_methods = []  # type: List
        self._precision = [ModelPrecision.FP32]
        self._optimization_type = ModelOptimizationType.MO

        # Create and initialize PyTorch model.
        self._model = self._load_model(task_environment.model)

        # Extra control variables.
        self._training_work_dir = None
        self._is_training = False
        self._should_stop = False
        # Exit
        self._optimization_type = ModelOptimizationType.NNCF

    def _set_attributes_by_hyperparams(self):
        quantization = self._hyperparams.nncf_optimization.enable_quantization
        pruning = self._hyperparams.nncf_optimization.enable_pruning
        if quantization and pruning:
            self._nncf_preset = "nncf_quantization_pruning"
            self._optimization_methods = [OptimizationMethod.QUANTIZATION, OptimizationMethod.FILTER_PRUNING]
            self._precision = [ModelPrecision.INT8]
            return
        if quantization and not pruning:
            self._nncf_preset = "nncf_quantization"
            self._optimization_methods = [OptimizationMethod.QUANTIZATION]
            self._precision = [ModelPrecision.INT8]
            return
        if not quantization and pruning:
            self._nncf_preset = "nncf_pruning"
            self._optimization_methods = [OptimizationMethod.FILTER_PRUNING]
            self._precision = [ModelPrecision.FP32]
            return
        raise RuntimeError("Not selected optimization algorithm")

    def _load_model(self, model: Optional[ModelEntity]):
        # NNCF parts
        nncf_config_path = os.path.join(self._base_dir, "compression_config.json")

        with open(nncf_config_path, encoding="UTF-8") as nncf_config_file:
            common_nncf_config = json.load(nncf_config_file)

        self._set_attributes_by_hyperparams()

        optimization_config = compose_nncf_config(common_nncf_config, [self._nncf_preset])

        max_acc_drop = self._hyperparams.nncf_optimization.maximal_accuracy_degradation / 100
        if "accuracy_aware_training" in optimization_config["nncf_config"]:
            # Update maximal_absolute_accuracy_degradation
            (
                optimization_config["nncf_config"]["accuracy_aware_training"]["params"][
                    "maximal_absolute_accuracy_degradation"
                ]
            ) = max_acc_drop
            # Force evaluation interval
            self._config.evaluation.interval = 1
        else:
            logger.info("NNCF config has no accuracy_aware_training parameters")

        self._config.update(optimization_config)

        compression_ctrl = None
        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))

            model = self._create_model(self._config, from_scratch=True)
            try:
                if is_state_nncf(model_data):
                    compression_ctrl, model = wrap_nncf_model(model, self._config, init_state_dict=model_data)
                    logger.info("Loaded model weights from Task Environment and wrapped by NNCF")
                else:
                    try:
                        load_state_dict(model, model_data["model"])
                        logger.info("Loaded model weights from Task Environment")
                        logger.info(f"Model architecture: {self._model_name}")
                    except BaseException as ex:
                        raise ValueError("Could not load the saved model. The model file structure is invalid.") from ex

                logger.info("Loaded model weights from Task Environment")
                logger.info(f"Model architecture: {self._model_name}")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") from ex
        else:
            raise ValueError("No trained model in project. NNCF require pretrained weights to compress the model")

        self._compression_ctrl = compression_ctrl
        return model

    def _create_compressed_model(self, dataset, config):
        init_dataloader = build_dataloader(
            dataset,
            config.data.samples_per_gpu,
            config.data.workers_per_gpu,
            len(config.gpu_ids),
            dist=False,
            seed=config.seed,
        )
        is_acc_aware_training_set = is_accuracy_aware_training_set(config.get("nncf_config"))

        if is_acc_aware_training_set:
            self._val_dataloader = build_val_dataloader(config, False)

        self._compression_ctrl, self._model = wrap_nncf_model(
            self._model,
            config,
            val_dataloader=self._val_dataloader,
            dataloader_for_init=init_dataloader,
            is_accuracy_aware=is_acc_aware_training_set,
        )

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        """NNCF Optimization."""
        if optimization_type is not OptimizationType.NNCF:
            raise RuntimeError("NNCF is the only supported optimization")

        train_dataset = dataset.get_subset(Subset.TRAINING)
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        config = self._config

        if optimization_parameters is not None:
            update_progress_callback = optimization_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback

        time_monitor = OptimizationProgressCallback(
            update_progress_callback, loading_stage_progress_percentage=5, initialization_stage_progress_percentage=5
        )
        learning_curves = DefaultDict(OTXLoggerHook.Curve)  # type: DefaultDict
        training_config = prepare_for_training(config, train_dataset, val_dataset, time_monitor, learning_curves)

        self._training_work_dir = training_config.work_dir
        mm_train_dataset = build_dataset(training_config.data.train)

        # Initialize NNCF parts if start from not compressed model
        if not self._compression_ctrl:
            self._create_compressed_model(mm_train_dataset, training_config)

        time_monitor.on_initialization_end()

        self._is_training = True
        self._model.train()

        train_segmentor(
            model=self._model,
            dataset=mm_train_dataset,
            cfg=training_config,
            validate=True,
            compression_ctrl=self._compression_ctrl,
        )

        self.save_model(output_model)

        output_model.model_format = ModelFormat.BASE_FRAMEWORK
        output_model.optimization_type = self._optimization_type
        output_model.optimization_methods = self._optimization_methods
        output_model.precision = self._precision

        self._is_training = False

    @check_input_parameters_type()
    def export(self, export_type: ExportType, output_model: ModelEntity):
        """NNCF Export Function."""
        if self._compression_ctrl is None:
            super().export(export_type, output_model)
        else:
            self._compression_ctrl.prepare_for_export()
            self._model.disable_dynamic_graph_building()
            super().export(export_type, output_model)
            self._model.enable_dynamic_graph_building()

    @check_input_parameters_type()
    def save_model(self, output_model: ModelEntity):
        """Save a model function."""
        buffer = io.BytesIO()
        hyperparams = self._task_environment.get_hyper_parameters(SegmentationConfig)  # type: ConfigDict
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        if not self._compression_ctrl:
            raise RuntimeError("Not found _compression_ctrl")
        modelinfo = {
            "compression_state": self._compression_ctrl.get_compression_state(),
            "meta": {
                "config": self._config,
                "nncf_enable_compression": True,
            },
            "model": self._model.state_dict(),
            "config": hyperparams_str,
            "labels": labels,
            "VERSION": 1,
        }

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))

    @staticmethod
    def _create_model(config: Config, from_scratch: bool = False):
        """Creates a model, based on the configuration in config.

        :param config: mmsegmentation configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights

        :return model: ModelEntity in training mode
        """

        model_cfg = copy.deepcopy(config.model)

        init_from = None if from_scratch else config.get("load_from", None)
        logger.warning(f"Init from: {init_from}")

        if init_from is not None:
            # No need to initialize backbone separately, if all weights are provided.
            model_cfg.pretrained = None
            logger.warning("build segmentor")
            model = build_segmentor(model_cfg)

            # Load all weights.
            logger.warning("load checkpoint")
            load_checkpoint(model, init_from, map_location="cpu", strict=False)
        else:
            logger.warning("build segmentor")
            model = build_segmentor(model_cfg)

        return model
