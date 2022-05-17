# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import io
import logging
import math
from typing import Dict
from typing import Optional

import os
import torch

import torchreid
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.model import (ModelEntity, ModelFormat, ModelOptimizationType, ModelPrecision,
                                    OptimizationMethod)
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import default_progress_callback
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import IOptimizationTask, OptimizationType
from ote_sdk.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from scripts.default_config import imagedata_kwargs, lr_scheduler_kwargs, optimizer_kwargs
from torchreid.apis.training import run_training
from torchreid.integration.nncf.compression import check_nncf_is_enabled, is_nncf_state, wrap_nncf_model
from torchreid.integration.nncf.compression_script_utils import (calculate_lr_for_nncf_training,
                                                                 patch_config)
from torchreid_tasks.inference_task import OTEClassificationInferenceTask
from torchreid_tasks.monitors import DefaultMetricsMonitor
from torchreid_tasks.utils import OTEClassificationDataset, OptimizationProgressCallback
from torchreid.ops import DataParallel
from torchreid.utils import set_random_seed, set_model_attr

logger = logging.getLogger(__name__)


class OTEClassificationNNCFTask(OTEClassificationInferenceTask, IOptimizationTask):

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for compressing classification models using NNCF.
        """
        logger.info('Loading OTEClassificationNNCFTask.')
        super().__init__(task_environment)

        check_nncf_is_enabled()

        # Set hyperparameters
        self._nncf_preset = None
        self._max_acc_drop = None
        self._set_attributes_by_hyperparams()

        # Patch the config
        if not self._cfg.nncf.nncf_config_path:
            self._cfg.nncf.nncf_config_path = os.path.join(self._base_dir, 'compression_config.json')
        self._cfg = patch_config(self._cfg, self._nncf_preset, self._max_acc_drop)

        self._compression_ctrl = None
        self._nncf_metainfo = None

        # Load NNCF model.
        if task_environment.model.optimization_type == ModelOptimizationType.NNCF:
            logger.info('Loading the NNCF model')
            self._compression_ctrl, self._model, self._nncf_metainfo = self._load_nncf_model(task_environment.model)

        # Set default model attributes.
        self._optimization_type = ModelOptimizationType.NNCF
        logger.info('OTEClassificationNNCFTask initialization completed')
        set_model_attr(self._model, 'mix_precision', self._cfg.train.mix_precision)

    @property
    def _initial_lr(self):
        return getattr(self, '__initial_lr')

    @_initial_lr.setter
    def _initial_lr(self, value):
        setattr(self, '__initial_lr', value)

    def _set_attributes_by_hyperparams(self):
        logger.info('Hyperparameters: ')
        logger.info(f'maximal_accuracy_degradation = '
                    f'{self._hyperparams.nncf_optimization.maximal_accuracy_degradation}')
        logger.info(f'enable_quantization = {self._hyperparams.nncf_optimization.enable_quantization}')
        logger.info(f'enable_pruning = {self._hyperparams.nncf_optimization.enable_pruning}')
        self._max_acc_drop = self._hyperparams.nncf_optimization.maximal_accuracy_degradation / 100.0
        quantization = self._hyperparams.nncf_optimization.enable_quantization
        pruning = self._hyperparams.nncf_optimization.enable_pruning
        if quantization and pruning:
            self._nncf_preset = 'nncf_quantization_pruning'
            self._optimization_methods = [OptimizationMethod.QUANTIZATION, OptimizationMethod.FILTER_PRUNING]
            self._precision = [ModelPrecision.INT8]
            return
        if quantization and not pruning:
            self._nncf_preset = 'nncf_quantization'
            self._optimization_methods = [OptimizationMethod.QUANTIZATION]
            self._precision = [ModelPrecision.INT8]
            return
        if not quantization and pruning:
            self._nncf_preset = 'nncf_pruning'
            self._optimization_methods = [OptimizationMethod.FILTER_PRUNING]
            self._precision = [ModelPrecision.FP32]
            return
        raise RuntimeError('Not selected optimization algorithm')

    def _load_model(self, model: ModelEntity, device: torch.device, pretrained_dict: Optional[Dict] = None):
        if model is None:
            raise ValueError('No trained model in the project. NNCF require pretrained weights to compress the model')

        if model.optimization_type == ModelOptimizationType.NNCF:
            logger.info('Skip loading the original model')
            return None

        model_data = pretrained_dict if pretrained_dict else self._load_model_data(model, 'weights.pth')
        if is_nncf_state(model_data):
            raise ValueError('Model optimization type is not consistent with the model checkpoint.')

        self._initial_lr = model_data.get('initial_lr')

        return super()._load_model(model, device, pretrained_dict=model_data)

    def _load_nncf_model(self, model: ModelEntity):
        if model is None:
            raise ValueError('No NNCF trained model in project.')

        model_data = self._load_model_data(model, 'weights.pth')
        if not is_nncf_state(model_data):
            raise ValueError('Model optimization type is not consistent with the NNCF model checkpoint.')
        model = self._create_model(self._cfg, from_scratch=True)

        compression_ctrl, model, nncf_metainfo = wrap_nncf_model(model, self._cfg, checkpoint_dict=model_data)
        logger.info('Loaded NNCF model weights from Task Environment.')
        return compression_ctrl, model, nncf_metainfo

    def _load_aux_models_data(self, model: ModelEntity):
        aux_models_data = []
        num_aux_models = len(self._cfg.mutual_learning.aux_configs)
        for idx in range(num_aux_models):
            data_name = f'aux_model_{idx + 1}.pth'
            if data_name not in model.model_adapters:
                return []
            model_data = self._load_model_data(model, data_name)
            aux_models_data.append(model_data)
        return aux_models_data

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        """ Optimize a model on a dataset """
        if optimization_type is not OptimizationType.NNCF:
            raise RuntimeError('NNCF is the only supported optimization')
        if self._compression_ctrl:
            raise RuntimeError('The model is already optimized. NNCF requires the original model for optimization.')
        if self._cfg.lr_finder.enable:
            raise RuntimeError('LR finder could not be used together with NNCF compression')

        aux_pretrained_dicts = self._load_aux_models_data(self._task_environment.model)
        num_aux_models = len(self._cfg.mutual_learning.aux_configs)
        num_aux_pretrained_dicts = len(aux_pretrained_dicts)
        if num_aux_models != num_aux_pretrained_dicts:
            raise RuntimeError('The pretrained weights are not provided for all aux models.')

        if optimization_parameters is not None:
            update_progress_callback = optimization_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback

        num_epoch = self._cfg.nncf_config['accuracy_aware_training']['params']['maximal_total_epochs']
        train_subset = dataset.get_subset(Subset.TRAINING)
        time_monitor = OptimizationProgressCallback(update_progress_callback,
                                                    num_epoch=num_epoch,
                                                    num_train_steps=max(1, math.floor(len(train_subset) /
                                                                                      self._cfg.train.batch_size)),
                                                    num_val_steps=0, num_test_steps=0,
                                                    loading_stage_progress_percentage=5,
                                                    initialization_stage_progress_percentage=5)

        self.metrics_monitor = DefaultMetricsMonitor()
        self.stop_callback.reset()

        set_random_seed(self._cfg.train.seed)
        val_subset = dataset.get_subset(Subset.VALIDATION)
        self._cfg.custom_datasets.roots = [OTEClassificationDataset(train_subset, self._labels, self._multilabel,
                                                                    self._hierarchical, self._multihead_class_info,
                                                                    keep_empty_label=self._empty_label in self._labels),
                                           OTEClassificationDataset(val_subset, self._labels, self._multilabel,
                                                                    self._hierarchical, self._multihead_class_info,
                                                                    keep_empty_label=self._empty_label in self._labels)]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self._cfg))

        self._compression_ctrl, self._model, self._nncf_metainfo = \
            wrap_nncf_model(self._model, self._cfg, datamanager_for_init=datamanager)

        time_monitor.on_initialization_end()

        self._cfg.train.lr = calculate_lr_for_nncf_training(self._cfg, self._initial_lr, False)

        train_model = self._model
        if self._cfg.use_gpu:
            main_device_ids = list(range(self.num_devices))
            extra_device_ids = [main_device_ids for _ in range(num_aux_models)]
            train_model = DataParallel(train_model, device_ids=main_device_ids,
                                       output_device=0).cuda(main_device_ids[0])
        else:
            extra_device_ids = [None for _ in range(num_aux_models)]

        optimizer = torchreid.optim.build_optimizer(train_model, **optimizer_kwargs(self._cfg))

        scheduler = torchreid.optim.build_lr_scheduler(optimizer, num_iter=datamanager.num_iter,
                                                       **lr_scheduler_kwargs(self._cfg))

        logger.info('Start training')
        time_monitor.on_train_begin()
        run_training(self._cfg, datamanager, train_model, optimizer,
                     scheduler, extra_device_ids, self._cfg.train.lr,
                     should_freeze_aux_models=True,
                     aux_pretrained_dicts=aux_pretrained_dicts,
                     tb_writer=self.metrics_monitor,
                     perf_monitor=time_monitor,
                     stop_callback=self.stop_callback,
                     nncf_metainfo=self._nncf_metainfo,
                     compression_ctrl=self._compression_ctrl)
        time_monitor.on_train_end()

        self.metrics_monitor.close()
        if self.stop_callback.check_stop():
            logger.info('Training cancelled.')
            return

        logger.info('Training completed')

        self.save_model(output_model)

        output_model.model_format = ModelFormat.BASE_FRAMEWORK
        output_model.optimization_type = self._optimization_type
        output_model.optimization_methods = self._optimization_methods
        output_model.precision = self._precision

    @check_input_parameters_type()
    def save_model(self, output_model: ModelEntity):
        state_dict = None
        if self._compression_ctrl is not None:
            state_dict = {
                'compression_state': self._compression_ctrl.get_compression_state(),
                'nncf_metainfo': self._nncf_metainfo
            }
        self._save_model(output_model, state_dict)

    @check_input_parameters_type()
    def export(self, export_type: ExportType, output_model: ModelEntity):
        if self._compression_ctrl is None:
            super().export(export_type, output_model)
        else:
            self._compression_ctrl.prepare_for_export()
            self._model.disable_dynamic_graph_building()
            super().export(export_type, output_model)
            self._model.enable_dynamic_graph_building()

    @staticmethod
    def _load_model_data(model, data_name):
        buffer = io.BytesIO(model.get_data(data_name))
        return torch.load(buffer, map_location=torch.device('cpu'))
