# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
import io
import os
import shutil
import tempfile
from typing import Optional, Union
import numpy as np
import torch
from mmcv.utils.config import Config, ConfigDict
from mpa.builder import build
from mpa.modules.hooks.cancel_interface_hook import CancelInterfaceHook
from mpa.stage import Stage
from mpa.utils.config_utils import update_or_add_custom_hook
from mpa.utils.logger import get_logger
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model import ModelEntity, ModelPrecision
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import (TrainParameters,
                                               UpdateProgressCallback)
from ote_sdk.serialization.label_mapper import LabelSchemaMapper


logger = get_logger()


class BaseTask:
    def __init__(self, task_config, task_environment: TaskEnvironment):
        self._task_config = task_config
        self._task_environment = task_environment
        self._hyperparams = task_environment.get_hyper_parameters(self._task_config)
        self._model_name = task_environment.model_template.name
        self._labels = task_environment.get_labels(include_empty=False)
        self._output_path = tempfile.mkdtemp(prefix='MPA-task-')
        logger.info(f'created output path at {self._output_path}')
        self.confidence_threshold = self._get_confidence_threshold(self._hyperparams)
        # Set default model attributes.
        self._model_label_schema = []
        self._optimization_methods = []
        self._precision = [ModelPrecision.FP32]
        self._model_ckpt = None
        if task_environment.model is not None:
            logger.info('loading the model from the task env.')
            state_dict = self._load_model_state_dict(self._task_environment.model)
            if state_dict:
                self._model_ckpt = os.path.join(self._output_path, 'env_model_ckpt.pth')
                if os.path.exists(self._model_ckpt):
                    os.remove(self._model_ckpt)
                torch.save(state_dict, self._model_ckpt)
                self._model_label_schema = self._load_model_label_schema(self._task_environment.model)

        # property below will be initialized by initialize()
        self._recipe_cfg = None
        self._stage_module = None
        self._model_cfg = None
        self._data_cfg = None
        self._mode = None
        self._time_monitor = None
        self._learning_curves = None
        self._is_training = False
        self._should_stop = False
        self.cancel_interface = None
        self.reserved_cancel = False
        self.on_hook_initialized = self.OnHookInitialized(self)

    def _run_task(self, stage_module, mode=None, dataset=None, parameters=None, **kwargs):
        self._initialize(dataset)
        # update model config -> model label schema
        data_classes = [label.name for label in self._labels]
        model_classes = [label.name for label in self._model_label_schema]
        self._model_cfg['model_classes'] = model_classes
        if dataset is not None:
            train_data_cfg = Stage.get_train_data_cfg(self._data_cfg)
            train_data_cfg['data_classes'] = data_classes
            new_classes = np.setdiff1d(data_classes, model_classes).tolist()
            train_data_cfg['old_new_indices'] = self._get_old_new_indices(dataset, new_classes)

        logger.info(f'running task... kwargs = {kwargs}')
        if self._recipe_cfg is None:
            raise RuntimeError(
                "'recipe_cfg' is not initialized yet."
                "call prepare() method before calling this method")

        if mode is not None:
            self._mode = mode

        common_cfg = ConfigDict(dict(output_path=self._output_path))

        # build workflow using recipe configuration
        workflow = build(self._recipe_cfg, self._mode, stage_type=stage_module, common_cfg=common_cfg)

        # run workflow with task specific model config and data config
        output = workflow.run(
            model_cfg=self._model_cfg,
            data_cfg=self._data_cfg,
            ir_path=None,
            model_ckpt=self._model_ckpt,
            mode=self._mode,
            **kwargs
        )
        logger.info('run task done.')
        return output

    def finalize(self):
        if self._recipe_cfg is not None:
            if self._recipe_cfg.get('cleanup_outputs', False):
                if os.path.exists(self._output_path):
                    shutil.rmtree(self._output_path, ignore_errors=False)

    def _delete_scratch_space(self):
        """
        Remove model checkpoints and mpa logs
        """

        if os.path.exists(self._output_path):
            shutil.rmtree(self._output_path, ignore_errors=False)

    def __del__(self):
        self.finalize()

    def _pre_task_run(self):
        pass

    @property
    def model_name(self):
        return self._task_environment.model_template.name

    @property
    def labels(self):
        return self._task_environment.get_labels(False)

    @property
    def template_file_path(self):
        return self._task_environment.model_template.model_template_path

    @property
    def hyperparams(self):
        return self._hyperparams

    def _initialize(self, dataset, output_model=None):
        """ prepare configurations to run a task through MPA's stage
        """
        logger.info('initializing....')
        self._init_recipe()
        recipe_hparams = self._init_recipe_hparam()
        if len(recipe_hparams) > 0:
            self._recipe_cfg.merge_from_dict(recipe_hparams)

        # prepare model config
        self._model_cfg = self._init_model_cfg()

        # add Cancel tranining hook
        update_or_add_custom_hook(self._recipe_cfg, ConfigDict(
            type='CancelInterfaceHook', init_callback=self.on_hook_initialized))
        if self._time_monitor is not None:
            update_or_add_custom_hook(self._recipe_cfg, ConfigDict(
                type='OTEProgressHook', time_monitor=self._time_monitor, verbose=True))
        if self._learning_curves is not None:
            self._recipe_cfg.log_config.hooks.append(
                {'type': 'OTELoggerHook', 'curves': self._learning_curves}
            )

        logger.info('initialized.')

    @abc.abstractmethod
    def _init_recipe(self):
        """
        initialize the MPA's target recipe. (inclusive of stage type)
        """
        raise NotImplementedError('this method should be implemented')

    def _init_model_cfg(self) -> Union[Config, None]:
        """
        initialize model_cfg for override recipe's model configuration.
        it can be None. (MPA's workflow consumable)
        """
        return None

    def _init_train_data_cfg(self, dataset: DatasetEntity) -> Union[Config, None]:
        """
        initialize data_cfg for override recipe's data configuration.
        it can be Config or None. (MPA's workflow consumable)
        """
        return None

    def _init_test_data_cfg(self, dataset: DatasetEntity) -> Union[Config, None]:
        """
        initialize data_cfg for override recipe's data configuration.
        it can be Config or None. (MPA's workflow consumable)
        """
        return None

    def _init_recipe_hparam(self) -> dict:
        """
        initialize recipe hyperparamter as dict.
        """
        return dict()

    def _load_model_state_dict(self, model: ModelEntity):
        if 'weights.pth' in model.model_adapters:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device('cpu'))

            # set confidence_threshold as well
            self.confidence_threshold = model_data.get('confidence_threshold', self.confidence_threshold)

            return model_data['model']
        else:
            return None

    def _load_model_label_schema(self, model: ModelEntity):
        # If a model has been trained and saved for the task already, create empty model and load weights here
        if "label_schema.json" in model.model_adapters:
            import json
            buffer = json.loads(model.get_data("label_schema.json").decode('utf-8'))
            model_label_schema = LabelSchemaMapper().backward(buffer)
            return model_label_schema.get_labels(include_empty=False)
        else:
            return self._labels

    def _get_old_new_indices(self, dataset, new_classes):
        ids_old, ids_new = [], []
        _dataset_label_schema_map = {label.name: label for label in self._labels}
        new_classes = [_dataset_label_schema_map[new_class] for new_class in new_classes]
        for i, item in enumerate(dataset.get_subset(Subset.TRAINING)):
            if item.annotation_scene.contains_any(new_classes):
                ids_new.append(i)
            else:
                ids_old.append(i)
        return {'old': ids_old, 'new': ids_new}

    @staticmethod
    def _get_confidence_threshold(hyperparams):
        confidence_threshold = 0.3
        if hasattr(hyperparams, 'postprocessing') and hasattr(hyperparams.postprocessing, 'confidence_threshold'):
            confidence_threshold = hyperparams.postprocessing.confidence_threshold
        return confidence_threshold

    def cancel_hook_initialized(self, cancel_interface: CancelInterfaceHook):
        logger.info('cancel hook is initialized')
        self.cancel_interface = cancel_interface
        if self.reserved_cancel:
            self.cancel_interface.cancel()

    class OnHookInitialized:
        def __init__(self, task_instance):
            self.task_instance = task_instance

        def __call__(self, cancel_interface):
            self.task_instance.cancel_hook_initialized(cancel_interface)

        def __repr__(self):
            return f"'{__name__}.OnHookInitialized'"

        def __reduce__(self):
            return (self.__class__, (id(self.task_instance),))
