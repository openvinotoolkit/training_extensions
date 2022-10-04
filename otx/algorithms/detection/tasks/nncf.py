"""NNCF Task of OTX Detection."""

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
import os
import tempfile
from typing import DefaultDict, List, Optional

import torch
from mmcv.runner import load_checkpoint, load_state_dict
from mmcv.utils import Config, ConfigDict
from mmdet.apis import train_detector
from mmdet.apis.fake_input import get_fake_input
from mmdet.apis.train import build_val_dataloader
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.integration.nncf import (
    check_nncf_is_enabled,
    is_accuracy_aware_training_set,
    is_state_nncf,
    wrap_nncf_model,
)
from mmdet.integration.nncf.config import compose_nncf_config
from mmdet.models import build_detector
from mpa.utils.config_utils import remove_custom_hook
from mpa.utils.logger import get_logger

from otx.algorithms.common.utils.hooks import OTELoggerHook
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.utils.config_utils import (
    patch_config,
    prepare_for_training,
    remove_from_config,
    set_hyperparams,
)
from otx.algorithms.detection.utils.otx_utils import OptimizationProgressCallback
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
from otx.api.entities.model_template import (
    parse_model_template,
    task_type_to_label_domain,
)
from otx.api.entities.optimization_parameters import (
    OptimizationParameters,
    default_progress_callback,
)
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

from .inference import DetectionInferenceTask

logger = get_logger()


class DetectionNNCFTask(DetectionInferenceTask, IOptimizationTask):
    """Task for compressing detection models using NNCF."""

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        # TODO: OTENNCFTASK + MPANNCFTASK, need to check base_model_path
        super().__init__(task_environment)
        curr_model_path = task_environment.model_template.model_template_path
        base_model_path = os.path.join(
            os.path.dirname(os.path.abspath(curr_model_path)),
            task_environment.model_template.base_model_path,
        )
        if os.path.isfile(base_model_path):
            logger.info(f"Base model for NNCF: {base_model_path}")
            task_environment.model_template = parse_model_template(base_model_path)
        self._val_dataloader = None
        self._compression_ctrl = None
        self._nncf_preset = "nncf_quantization"
        check_nncf_is_enabled()
        # super().__init__(task_environment)
        self._task_environment = task_environment
        self._task_type = task_environment.model_template.task_type
        self._output_path = tempfile.mkdtemp(prefix="otx-det-scratch-")
        logger.info(f"Scratch space created at {self._output_path}")

        self._model_name = task_environment.model_template.name
        self._labels = task_environment.get_labels(False)

        template_file_path = task_environment.model_template.model_template_path

        # Get and prepare mmdet config.
        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))

        # Align MPA config for nncf task
        self._initialize()
        self._config = Config()
        if self._recipe_cfg:
            self._config = self._recipe_cfg
            self._config.merge_from_dict(self._model_cfg)
            self._config.data.pop("super_type", None)
            remove_custom_hook(self._config, "CancelInterfaceHook")
        else:
            config_file_path = os.path.join(self._base_dir, "model.py")
            self._config = Config.fromfile(config_file_path)

        patch_config(
            self._config,
            self._output_path,
            self._labels,
            task_type_to_label_domain(self._task_type),
            random_seed=42,
        )
        set_hyperparams(self._config, self._hyperparams)
        self.confidence_threshold: float = self._hyperparams.postprocessing.confidence_threshold

        # Set default model attributes.
        self._optimization_methods = []  # type: List
        self._precision = self._precision_from_config
        self._optimization_type = ModelOptimizationType.MO

        # Create and initialize PyTorch model.
        logger.info("Loading the model")
        self._model = self._load_model(task_environment.model)

        # Extra control variables.
        self._training_work_dir = None
        self._is_training = False
        self._should_stop = False
        logger.info("Task initialization completed")
        self._optimization_type = ModelOptimizationType.NNCF

    def _set_attributes_by_hyperparams(self):
        quantization = self._hyperparams.nncf_optimization.enable_quantization
        pruning = self._hyperparams.nncf_optimization.enable_pruning
        if quantization and pruning:
            self._nncf_preset = "nncf_quantization_pruning"
            self._optimization_methods = [
                OptimizationMethod.QUANTIZATION,
                OptimizationMethod.FILTER_PRUNING,
            ]
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
            self._precision = self._precision_from_config
            return
        raise RuntimeError("Not selected optimization algorithm")

    def _load_model(self, model: Optional[ModelEntity]):
        # NNCF parts
        nncf_config_path = os.path.join(self._base_dir, "compression_config.json")

        with open(nncf_config_path) as nncf_config_file:
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

            self.confidence_threshold = model_data.get(
                "confidence_threshold",
                self._hyperparams.postprocessing.confidence_threshold,
            )
            if model_data.get("anchors"):
                anchors = model_data["anchors"]
                self._config.model.bbox_head.anchor_generator.heights = anchors["heights"]
                self._config.model.bbox_head.anchor_generator.widths = anchors["widths"]

            model = self._create_model(self._config, from_scratch=True)
            try:
                if is_state_nncf(model_data):
                    compression_ctrl, model = wrap_nncf_model(
                        model,
                        self._config,
                        init_state_dict=model_data,
                        get_fake_input_func=get_fake_input,
                    )
                    logger.info("Loaded model weights from Task Environment and wrapped by NNCF")
                else:
                    try:
                        load_state_dict(model, model_data["model"])

                        # It prevent model from being overwritten
                        # if "load_from" in self._config:
                        #    self._config.load_from = None

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

    @staticmethod
    def _create_model(config: Config, from_scratch: bool = False):
        """Creates a model, based on the configuration in config.

        :param config: mmdetection configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights

        :return model: ModelEntity in training mode
        """
        model_cfg = copy.deepcopy(config.model)

        init_from = None if from_scratch else config.get("load_from", None)
        logger.warning(init_from)
        if init_from is not None:
            # No need to initialize backbone separately, if all weights are provided.
            model_cfg.pretrained = None
            logger.warning("build detector")
            model = build_detector(model_cfg)
            # Load all weights.
            logger.warning("load checkpoint")
            load_checkpoint(model, init_from, map_location="cpu")
        else:
            logger.warning("build detector")
            model = build_detector(model_cfg)
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
            get_fake_input_func=get_fake_input,
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
            update_progress_callback,
            loading_stage_progress_percentage=5,
            initialization_stage_progress_percentage=5,
        )
        learning_curves = DefaultDict(OTELoggerHook.Curve)  # type: DefaultDict
        training_config = prepare_for_training(config, train_dataset, val_dataset, time_monitor, learning_curves)
        mm_train_dataset = build_dataset(training_config.data.train)

        if torch.cuda.is_available():
            self._model.cuda(training_config.gpu_ids[0])

        # Initialize NNCF parts if start from not compressed model
        if not self._compression_ctrl:
            self._create_compressed_model(mm_train_dataset, training_config)

        time_monitor.on_initialization_end()

        # Run training.
        self._training_work_dir = training_config.work_dir
        self._is_training = True
        self._model.train()

        fp16 = training_config.get("fp16", None)

        if fp16 is not None:
            remove_from_config(training_config, "fp16")
            logger.warn("fp16 option is not supported in NNCF. Switch to fp32.")

        train_detector(
            model=self._model,
            dataset=mm_train_dataset,
            cfg=training_config,
            validate=True,
            val_dataloader=self._val_dataloader,
            compression_ctrl=self._compression_ctrl,
        )

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return

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
        """Saving model function for NNCF Task."""
        buffer = io.BytesIO()
        hyperparams = self._task_environment.get_hyper_parameters(DetectionConfig)  # type: ConfigDict
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        # WA for scheduler resetting in NNCF
        if not self._compression_ctrl:
            raise RuntimeError("Not found _compression_ctrl")
        compression_state = self._compression_ctrl.get_compression_state()
        for algo_state in compression_state.get("ctrl_state", {}).values():
            if not algo_state.get("scheduler_state"):
                algo_state["scheduler_state"] = {"current_step": 0, "current_epoch": 0}
        modelinfo = {
            "compression_state": compression_state,
            "meta": {
                "config": self._config,
                "nncf_enable_compression": True,
            },
            "model": self._model.state_dict(),
            "config": hyperparams_str,
            "labels": labels,
            "confidence_threshold": self.confidence_threshold,
            "VERSION": 1,
        }

        if hasattr(self._config.model, "bbox_head") and hasattr(self._config.model.bbox_head, "anchor_generator"):
            if getattr(
                self._config.model.bbox_head.anchor_generator,
                "reclustering_anchors",
                False,
            ):
                generator = self._model.bbox_head.anchor_generator
                modelinfo["anchors"] = {
                    "heights": generator.heights,
                    "widths": generator.widths,
                }

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
