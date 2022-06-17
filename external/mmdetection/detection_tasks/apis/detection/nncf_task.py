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
import json
import os
from collections import defaultdict
from typing import Optional

import torch
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from ote_sdk.entities.optimization_parameters import default_progress_callback, OptimizationParameters
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.serialization.label_mapper import label_schema_to_bytes
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import IOptimizationTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from ote_sdk.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

from mmdet.apis import train_detector
from mmdet.apis.fake_input import get_fake_input
from detection_tasks.apis.detection.config_utils import prepare_for_training, remove_from_config
from detection_tasks.apis.detection.configuration import OTEDetectionConfig
from detection_tasks.apis.detection.inference_task import OTEDetectionInferenceTask
from detection_tasks.apis.detection.ote_utils import OptimizationProgressCallback
from detection_tasks.extension.utils.hooks import OTELoggerHook
from mmdet.apis.train import build_val_dataloader
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.integration.nncf import check_nncf_is_enabled
from mmdet.integration.nncf import is_state_nncf
from mmdet.integration.nncf import wrap_nncf_model
from mmdet.integration.nncf import is_accuracy_aware_training_set
from mmdet.integration.nncf.config import compose_nncf_config
from mmdet.utils.logger import get_root_logger


logger = get_root_logger()


class OTEDetectionNNCFTask(OTEDetectionInferenceTask, IOptimizationTask):

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for compressing object detection models using NNCF.
        """
        self._val_dataloader = None
        self._compression_ctrl = None
        self._nncf_preset = "nncf_quantization"
        check_nncf_is_enabled()
        super().__init__(task_environment)
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
            self._precision = self._precision_from_config
            return
        raise RuntimeError('Not selected optimization algorithm')

    def _load_model(self, model: ModelEntity):
        # NNCF parts
        nncf_config_path = os.path.join(self._base_dir, "compression_config.json")

        with open(nncf_config_path) as nncf_config_file:
            common_nncf_config = json.load(nncf_config_file)

        self._set_attributes_by_hyperparams()

        optimization_config = compose_nncf_config(common_nncf_config, [self._nncf_preset])

        max_acc_drop = self._hyperparams.nncf_optimization.maximal_accuracy_degradation / 100
        if "accuracy_aware_training" in optimization_config["nncf_config"]:
            # Update maximal_absolute_accuracy_degradation
            (optimization_config["nncf_config"]["accuracy_aware_training"]
                                ["params"]["maximal_absolute_accuracy_degradation"]) = max_acc_drop
            # Force evaluation interval
            self._config.evaluation.interval = 1
        else:
            logger.info("NNCF config has no accuracy_aware_training parameters")

        self._config.update(optimization_config)

        compression_ctrl = None
        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device('cpu'))

            self.confidence_threshold = model_data.get('confidence_threshold',
                self._hyperparams.postprocessing.confidence_threshold)
            if model_data.get('anchors'):
                anchors = model_data['anchors']
                self._config.model.bbox_head.anchor_generator.heights = anchors['heights']
                self._config.model.bbox_head.anchor_generator.widths = anchors['widths']

            model = self._create_model(self._config, from_scratch=True)
            try:
                if is_state_nncf(model_data):
                    compression_ctrl, model = wrap_nncf_model(
                        model,
                        self._config,
                        init_state_dict=model_data,
                        get_fake_input_func=get_fake_input
                    )
                    logger.info("Loaded model weights from Task Environment and wrapped by NNCF")
                else:
                    try:
                        model.load_state_dict(model_data['model'])
                        logger.info(f"Loaded model weights from Task Environment")
                        logger.info(f"Model architecture: {self._model_name}")
                    except BaseException as ex:
                        raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                            from ex

                logger.info(f"Loaded model weights from Task Environment")
                logger.info(f"Model architecture: {self._model_name}")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                    from ex
        else:
            raise ValueError(f"No trained model in project. NNCF require pretrained weights to compress the model")

        self._compression_ctrl = compression_ctrl
        return model

    def _create_compressed_model(self, dataset, config):
        init_dataloader = build_dataloader(
            dataset,
            config.data.samples_per_gpu,
            config.data.workers_per_gpu,
            len(config.gpu_ids),
            dist=False,
            seed=config.seed)
        is_acc_aware_training_set = is_accuracy_aware_training_set(config.get("nncf_config"))

        if is_acc_aware_training_set:
            self._val_dataloader = build_val_dataloader(config, False)

        self._compression_ctrl, self._model = wrap_nncf_model(
            self._model,
            config,
            val_dataloader=self._val_dataloader,
            dataloader_for_init=init_dataloader,
            get_fake_input_func=get_fake_input,
            is_accuracy_aware=is_acc_aware_training_set)

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        if optimization_type is not OptimizationType.NNCF:
            raise RuntimeError("NNCF is the only supported optimization")

        train_dataset = dataset.get_subset(Subset.TRAINING)
        val_dataset = dataset.get_subset(Subset.VALIDATION)

        config = self._config

        if optimization_parameters is not None:
            update_progress_callback = optimization_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback

        time_monitor = OptimizationProgressCallback(update_progress_callback,
                                                    loading_stage_progress_percentage=5,
                                                    initialization_stage_progress_percentage=5)
        learning_curves = defaultdict(OTELoggerHook.Curve)
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

        train_detector(model=self._model,
                       dataset=mm_train_dataset,
                       cfg=training_config,
                       validate=True,
                       val_dataloader=self._val_dataloader,
                       compression_ctrl=self._compression_ctrl)

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled
        if self._should_stop:
            logger.info('Training cancelled.')
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
        if self._compression_ctrl is None:
            super().export(export_type, output_model)
        else:
            self._compression_ctrl.prepare_for_export()
            self._model.disable_dynamic_graph_building()
            super().export(export_type, output_model)
            self._model.enable_dynamic_graph_building()

    @check_input_parameters_type()
    def save_model(self, output_model: ModelEntity):
        buffer = io.BytesIO()
        hyperparams = self._task_environment.get_hyper_parameters(OTEDetectionConfig)
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        # WA for scheduler resetting in NNCF
        compression_state = self._compression_ctrl.get_compression_state()
        for algo_state in compression_state.get('ctrl_state', {}).values():
            if not algo_state.get('scheduler_state'):
                algo_state['scheduler_state'] = {'current_step': 0, 'current_epoch': 0}
        modelinfo = {
            'compression_state': compression_state,
            'meta': {
                'config': self._config,
                'nncf_enable_compression': True,
            },
            'model': self._model.state_dict(),
            'config': hyperparams_str,
            'labels': labels,
            'confidence_threshold': self.confidence_threshold,
            'VERSION': 1,
        }

        if hasattr(self._config.model, 'bbox_head') and hasattr(self._config.model.bbox_head, 'anchor_generator'):
            if getattr(self._config.model.bbox_head.anchor_generator, 'reclustering_anchors', False):
                generator = self._model.bbox_head.anchor_generator
                modelinfo['anchors'] = {'heights': generator.heights, 'widths': generator.widths}

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))
