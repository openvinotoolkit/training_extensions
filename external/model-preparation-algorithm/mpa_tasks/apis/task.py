# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
import io
import os
import shutil
import tempfile
from typing import Union

import numpy as np
import torch
from mmcv.utils.config import Config, ConfigDict
from mpa.builder import build
from mpa.modules.hooks.cancel_interface_hook import CancelInterfaceHook
from mpa.stage import Stage
from mpa.utils.config_utils import remove_custom_hook, update_or_add_custom_hook
from mpa.utils.logger import get_logger
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.model import ModelEntity, ModelPrecision
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.serialization.label_mapper import LabelSchemaMapper

logger = get_logger()
DEFAULT_META_KEYS = (
    "filename",
    "ori_filename",
    "ori_shape",
    "img_shape",
    "pad_shape",
    "scale_factor",
    "flip",
    "flip_direction",
    "img_norm_cfg",
)


class BaseTask:
    def __init__(self, task_config, task_environment: TaskEnvironment):
        self._task_config = task_config
        self._task_environment = task_environment
        self._hyperparams = task_environment.get_hyper_parameters(self._task_config)
        self._model_name = task_environment.model_template.name
        self._task_type = task_environment.model_template.task_type
        self._labels = task_environment.get_labels(include_empty=False)
        self._output_path = tempfile.mkdtemp(prefix="MPA-task-")
        logger.info(f"created output path at {self._output_path}")
        self.confidence_threshold = self._get_confidence_threshold(self._hyperparams)
        # Set default model attributes.
        self._model_label_schema = []
        self._optimization_methods = []
        self._model_ckpt = None
        self._anchors = {}
        if task_environment.model is not None:
            logger.info("loading the model from the task env.")
            state_dict = self._load_model_state_dict(self._task_environment.model)
            if state_dict:
                self._model_ckpt = os.path.join(self._output_path, "env_model_ckpt.pth")
                if os.path.exists(self._model_ckpt):
                    os.remove(self._model_ckpt)
                torch.save(state_dict, self._model_ckpt)
                self._model_label_schema = self._load_model_label_schema(self._task_environment.model)

        # property below will be initialized by initialize()
        self._recipe_cfg = None
        self._stage_module = None
        self._model_cfg = None
        self._precision = None
        self._data_cfg = None
        self._mode = None
        self._time_monitor = None
        self._learning_curves = None
        self._is_training = False
        self._should_stop = False
        self.cancel_interface = None
        self.reserved_cancel = False
        self.on_hook_initialized = self.OnHookInitialized(self)

        # to override configuration at runtime
        self.override_configs = {}

    def _run_task(self, stage_module, mode=None, dataset=None, parameters=None, **kwargs):
        # FIXME: Temporary remedy for CVS-88098
        export = kwargs.get("export", False)
        self._initialize(dataset, export=export)
        # update model config -> model label schema
        data_classes = [label.name for label in self._labels]
        model_classes = [label.name for label in self._model_label_schema]
        self._model_cfg["model_classes"] = model_classes
        if dataset is not None:
            train_data_cfg = Stage.get_train_data_cfg(self._data_cfg)
            train_data_cfg["data_classes"] = data_classes
            new_classes = np.setdiff1d(data_classes, model_classes).tolist()
            train_data_cfg["new_classes"] = new_classes

        logger.info(f"running task... kwargs = {kwargs}")
        if self._recipe_cfg is None:
            raise RuntimeError(
                "'recipe_cfg' is not initialized yet." "call prepare() method before calling this method"
            )

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
            **kwargs,
        )
        logger.info("run task done.")
        return output

    def finalize(self):
        if self._recipe_cfg is not None:
            if self._recipe_cfg.get("cleanup_outputs", False):
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

    def _initialize(self, dataset=None, output_model=None, export=False):
        """prepare configurations to run a task through MPA's stage"""
        logger.info("initializing....")
        self._init_recipe()

        if not export:
            recipe_hparams = self._init_recipe_hparam()
            if len(recipe_hparams) > 0:
                self._recipe_cfg.merge_from_dict(recipe_hparams)

        if "custom_hooks" in self.override_configs:
            override_custom_hooks = self.override_configs.pop("custom_hooks")
            for override_custom_hook in override_custom_hooks:
                update_or_add_custom_hook(self._recipe_cfg, ConfigDict(override_custom_hook))
        if len(self.override_configs) > 0:
            logger.info(f"before override configs merging = {self._recipe_cfg}")
            self._recipe_cfg.merge_from_dict(self.override_configs)
            logger.info(f"after override configs merging = {self._recipe_cfg}")

        # prepare model config
        self._model_cfg = self._init_model_cfg()

        # Remove FP16 config if running on CPU device and revert to FP32
        # https://github.com/pytorch/pytorch/issues/23377
        if not torch.cuda.is_available() and "fp16" in self._model_cfg:
            logger.info("Revert FP16 to FP32 on CPU device")
            if isinstance(self._model_cfg, Config):
                del self._model_cfg._cfg_dict["fp16"]
            elif isinstance(self._model_cfg, ConfigDict):
                del self._model_cfg["fp16"]
        self._precision = [ModelPrecision.FP32]

        # Add/remove adaptive interval hook
        if self._recipe_cfg.get("use_adaptive_interval", False):
            self._recipe_cfg.adaptive_validation_interval = self._recipe_cfg.get(
                "adaptive_validation_interval", dict(max_interval=5)
            )
        else:
            self._recipe_cfg.pop("adaptive_validation_interval", None)

        # Add/remove early stop hook
        if "early_stop" in self._recipe_cfg:
            remove_custom_hook(self._recipe_cfg, "EarlyStoppingHook")
            early_stop = self._recipe_cfg.get("early_stop", False)
            if early_stop:
                early_stop_hook = ConfigDict(
                    type="LazyEarlyStoppingHook",
                    start=early_stop.start,
                    patience=early_stop.patience,
                    iteration_patience=early_stop.iteration_patience,
                    interval=1,
                    metric=self._recipe_cfg.early_stop_metric,
                    priority=75,
                )
                update_or_add_custom_hook(self._recipe_cfg, early_stop_hook)
            else:
                remove_custom_hook(self._recipe_cfg, "LazyEarlyStoppingHook")

        # add Cancel tranining hook
        update_or_add_custom_hook(
            self._recipe_cfg,
            ConfigDict(type="CancelInterfaceHook", init_callback=self.on_hook_initialized),
        )
        if self._time_monitor is not None:
            update_or_add_custom_hook(
                self._recipe_cfg,
                ConfigDict(
                    type="OTEProgressHook",
                    time_monitor=self._time_monitor,
                    verbose=True,
                    priority=71,
                ),
            )
        if self._learning_curves is not None:
            self._recipe_cfg.log_config.hooks.append({"type": "OTELoggerHook", "curves": self._learning_curves})

        logger.info("initialized.")

    @abc.abstractmethod
    def _init_recipe(self):
        """
        initialize the MPA's target recipe. (inclusive of stage type)
        """
        raise NotImplementedError("this method should be implemented")

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
        if "weights.pth" in model.model_adapters:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))

            # set confidence_threshold as well
            self.confidence_threshold = model_data.get("confidence_threshold", self.confidence_threshold)
            if model_data.get("anchors"):
                self._anchors = model_data["anchors"]

            return model_data.get("model", model_data.get("state_dict", None))
        else:
            return None

    def _load_model_label_schema(self, model: ModelEntity):
        # If a model has been trained and saved for the task already, create empty model and load weights here
        if "label_schema.json" in model.model_adapters:
            import json

            buffer = json.loads(model.get_data("label_schema.json").decode("utf-8"))
            model_label_schema = LabelSchemaMapper().backward(buffer)
            return model_label_schema.get_labels(include_empty=False)
        else:
            return self._labels

    @staticmethod
    def _get_meta_keys(pipeline_step):
        meta_keys = list(pipeline_step.get("meta_keys", DEFAULT_META_KEYS))
        meta_keys.append("ignored_labels")
        pipeline_step["meta_keys"] = set(meta_keys)
        return pipeline_step

    @staticmethod
    def _get_confidence_threshold(hyperparams):
        confidence_threshold = 0.3
        if hasattr(hyperparams, "postprocessing") and hasattr(hyperparams.postprocessing, "confidence_threshold"):
            confidence_threshold = hyperparams.postprocessing.confidence_threshold
        return confidence_threshold

    def cancel_hook_initialized(self, cancel_interface: CancelInterfaceHook):
        logger.info("cancel hook is initialized")
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

    def update_override_configurations(self, config):
        logger.info(f"update override config with: {config}")
        config = ConfigDict(**config)
        self.override_configs.update(config)
