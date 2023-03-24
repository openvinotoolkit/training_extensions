"""BaseTask for Classification/Detection/Segmentation."""

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


import abc
import io
import os
import shutil
import tempfile
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from mmcv.utils.config import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.hooks import OTXLoggerHook
from otx.algorithms.common.adapters.mmcv.hooks.cancel_hook import CancelInterfaceHook
from otx.algorithms.common.adapters.mmcv.utils import (
    align_data_config_with_recipe,
    get_configs_by_pairs,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    MPAConfig,
    add_custom_hook_if_not_exists,
    remove_custom_hook,
    update_or_add_custom_hook,
)
from otx.algorithms.common.configs import TrainType
from otx.algorithms.common.utils import UncopiableDefaultDict
from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.model import ModelEntity, ModelPrecision, OptimizationMethod
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import LabelSchemaMapper
from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.export_interface import IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload
from otx.api.utils.argument_checks import check_input_parameters_type
from otx.core.data import caching
from otx.mpa.builder import build
from otx.mpa.stage import Stage

logger = get_logger()
TRAIN_TYPE_DIR_PATH = {
    TrainType.Incremental.name: ".",
    TrainType.Selfsupervised.name: "selfsl",
    TrainType.Semisupervised.name: "semisl",
}


# pylint: disable=too-many-instance-attributes, protected-access
class BaseTask(IInferenceTask, IExportTask, IEvaluationTask, IUnload):
    """BaseTask for OTX Algorithms."""

    _task_environment: TaskEnvironment

    @check_input_parameters_type()
    def __init__(self, task_config, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        self._task_config = task_config
        self._task_environment = task_environment
        self._hyperparams = task_environment.get_hyper_parameters(self._task_config)  # type: ConfigDict
        self._model_name = task_environment.model_template.name
        self._task_type = task_environment.model_template.task_type
        self._labels = task_environment.get_labels(include_empty=False)
        self.confidence_threshold = self._get_confidence_threshold(self._hyperparams)
        # Set default model attributes.
        self._model_label_schema = []  # type: List[LabelEntity]
        self._optimization_methods = []  # type: List[OptimizationMethod]
        self._model_ckpt = None
        self._resume = False
        self._anchors = {}  # type: Dict[str, int]
        self._work_dir_is_temp = False
        if output_path is None:
            output_path = tempfile.mkdtemp(prefix="OTX-task-")
            self._work_dir_is_temp = True
        self._output_path = output_path
        logger.info(f"created output path at {self._output_path}")
        if task_environment.model is not None:
            logger.info("loading the model from the task env.")
            state_dict = self._load_model_ckpt(self._task_environment.model)
            if state_dict:
                self._model_ckpt = os.path.join(self._output_path, "env_model_ckpt.pth")
                if os.path.exists(self._model_ckpt):
                    os.remove(self._model_ckpt)
                torch.save(state_dict, self._model_ckpt)
                self._model_label_schema = self._load_model_label_schema(self._task_environment.model)
                self._resume = self._load_resume_info(self._task_environment.model)

        # property below will be initialized by initialize()
        self._recipe_cfg = None
        self._stage_module = None
        self._precision = [ModelPrecision.FP32]
        self._data_cfg = None
        self._mode = None
        self._time_monitor = None  # type: Optional[TimeMonitorCallback]
        self._learning_curves = UncopiableDefaultDict(OTXLoggerHook.Curve)
        self._is_training = False
        self._should_stop = False
        self.cancel_interface = None  # type: Optional[CancelInterfaceHook]
        self.reserved_cancel = False
        self.on_hook_initialized = self.OnHookInitialized(self)

        # Initialize Train type related var
        self._train_type = self._hyperparams.algo_backend.train_type
        self._model_dir = os.path.join(
            os.path.abspath(os.path.dirname(self.template_file_path)), TRAIN_TYPE_DIR_PATH[self._train_type.name]
        )

        # to override configuration at runtime
        self.override_configs = {}  # type: Dict[str, str]

    def _run_task(self, stage_module, mode=None, dataset=None, **kwargs):
        self._initialize(kwargs)
        stage_module = self._update_stage_module(stage_module)

        if mode is not None:
            self._mode = mode

        # deepcopy all configs to make sure
        # changes under MPA and below does not take an effect to OTX for clear distinction
        recipe_cfg = deepcopy(self._recipe_cfg)
        data_cfg = deepcopy(self._data_cfg)
        assert recipe_cfg is not None, "'recipe_cfg' is not initialized."

        # update model config -> model label schema
        data_classes = [label.name for label in self._labels]
        model_classes = [label.name for label in self._model_label_schema]
        recipe_cfg["model_classes"] = model_classes
        if dataset is not None:
            train_data_cfg = Stage.get_data_cfg(data_cfg, "train")
            train_data_cfg["data_classes"] = data_classes
            new_classes = np.setdiff1d(data_classes, model_classes).tolist()
            train_data_cfg["new_classes"] = new_classes

        logger.info(  # pylint: disable=logging-not-lazy
            "running task... kwargs = "
            + str({k: v if k != "model_builder" else object.__repr__(v) for k, v in kwargs.items()})
        )

        common_cfg = ConfigDict(dict(output_path=self._output_path, resume=self._resume))

        # build workflow using recipe configuration
        workflow = build(
            recipe_cfg,
            self._mode,
            stage_type=stage_module,
            common_cfg=common_cfg,
        )

        # run workflow with task specific model config and data config
        output = workflow.run(
            model_cfg=recipe_cfg,
            data_cfg=data_cfg,
            ir_model_path=None,
            ir_weight_path=None,
            model_ckpt=self._model_ckpt,
            mode=self._mode,
            **kwargs,
        )
        logger.info("run task done.")
        return output

    def _delete_scratch_space(self):
        """Remove model checkpoints and mpa logs."""
        if os.path.exists(self._output_path):
            shutil.rmtree(self._output_path, ignore_errors=False)

    def _pre_task_run(self):
        pass

    @property
    def project_path(self):
        """Return output path with logs."""
        return self._output_path

    @property
    def model_name(self):
        """Name of Model Template."""
        return self._task_environment.model_template.name

    @property
    def labels(self):
        """Label List of Task Environment."""
        return self._task_environment.get_labels(False)

    @property
    def template_file_path(self):
        """Model Template file path."""
        return self._task_environment.model_template.model_template_path

    @property
    def data_pipeline_path(self):
        """Base Data Pipeline file path."""
        # TODO: Temporarily use data_pipeline.py next to model.py.may change later.
        if self._hyperparams.tiling_parameters.enable_tiling:
            return os.path.join(self._model_dir, "tile_pipeline.py")
        return os.path.join(self._model_dir, "data_pipeline.py")

    @property
    def hyperparams(self):
        """Hyper Parameters configuration."""
        return self._hyperparams

    # pylint: disable-next=too-many-branches,too-many-statements
    def _initialize(self, options=None):  # noqa: C901
        """Prepare configurations to run a task through MPA's stage."""
        if options is None:
            options = {}

        export = options.get("export", False)
        fp16_export = options.get("enable_fp16", False)

        logger.info("initializing....")
        self._init_recipe()

        if not export:
            # FIXME: Temporary remedy for CVS-88098
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

        # Remove FP16 config if running on CPU device and revert to FP32
        # https://github.com/pytorch/pytorch/issues/23377
        if not torch.cuda.is_available() and "fp16" in self._recipe_cfg:
            logger.info("Revert FP16 to FP32 on CPU device")
            if isinstance(self._recipe_cfg, Config):
                del self._recipe_cfg._cfg_dict["fp16"]
            elif isinstance(self._recipe_cfg, ConfigDict):
                del self._recipe_cfg["fp16"]

        # default adaptive hook for evaluating before and after training
        add_custom_hook_if_not_exists(
            self._recipe_cfg,
            ConfigDict(
                type="AdaptiveTrainSchedulingHook",
                enable_adaptive_interval_hook=False,
                enable_eval_before_run=True,
            ),
        )
        # Add/remove adaptive interval hook
        if self._recipe_cfg.get("use_adaptive_interval", False):
            update_or_add_custom_hook(
                self._recipe_cfg,
                ConfigDict(
                    {
                        "type": "AdaptiveTrainSchedulingHook",
                        "max_interval": 5,
                        "enable_adaptive_interval_hook": True,
                        "enable_eval_before_run": True,
                        **self._recipe_cfg.pop("adaptive_validation_interval", {}),
                    }
                ),
            )
        else:
            self._recipe_cfg.pop("adaptive_validation_interval", None)

        self.set_early_stopping_hook()

        # add Cancel tranining hook
        update_or_add_custom_hook(
            self._recipe_cfg,
            ConfigDict(type="CancelInterfaceHook", init_callback=self.on_hook_initialized),
        )
        if self._time_monitor is not None:
            update_or_add_custom_hook(
                self._recipe_cfg,
                ConfigDict(
                    type="OTXProgressHook",
                    time_monitor=self._time_monitor,
                    verbose=True,
                    priority=71,
                ),
            )
        self._recipe_cfg.log_config.hooks.append({"type": "OTXLoggerHook", "curves": self._learning_curves})

        # make sure model to be in a training mode even after model is evaluated (mmcv bug)
        update_or_add_custom_hook(
            self._recipe_cfg,
            ConfigDict(type="ForceTrainModeHook", priority="LOWEST"),
        )

        # if num_workers is 0, persistent_workers must be False
        data_cfg = self._recipe_cfg.data
        for subset in ["train", "val", "test", "unlabeled"]:
            if subset not in data_cfg:
                continue
            dataloader_cfg = data_cfg.get(f"{subset}_dataloader", ConfigDict())
            workers_per_gpu = dataloader_cfg.get(
                "workers_per_gpu",
                data_cfg.get("workers_per_gpu", 0),
            )
            if workers_per_gpu == 0:
                dataloader_cfg["persistent_workers"] = False
                data_cfg[f"{subset}_dataloader"] = dataloader_cfg

        # Update recipe with caching modules
        self._update_caching_modules(data_cfg)

        if self._data_cfg is not None:
            align_data_config_with_recipe(self._data_cfg, self._recipe_cfg)

        if export:
            if fp16_export:
                self._precision[0] = ModelPrecision.FP16
            options["deploy_cfg"] = self._init_deploy_cfg()
            if options.get("precision", None) is None:
                assert len(self._precision) == 1
                options["precision"] = str(self._precision[0])

            options["deploy_cfg"]["dump_features"] = options["dump_features"]
            if options["dump_features"]:
                output_names = options["deploy_cfg"]["ir_config"]["output_names"]
                if "feature_vector" not in output_names:
                    options["deploy_cfg"]["ir_config"]["output_names"].append("feature_vector")
                if options["deploy_cfg"]["codebase_config"]["task"] != "Segmentation":
                    if "saliency_map" not in output_names:
                        options["deploy_cfg"]["ir_config"]["output_names"].append("saliency_map")

        self._initialize_post_hook(options)

        logger.info("initialized.")

    def _initialize_post_hook(self, options=None):
        pass

    @abc.abstractmethod
    def _init_recipe(self):
        """Initialize the MPA's target recipe (inclusive of stage type)."""
        raise NotImplementedError("this method should be implemented")

    def _init_model_cfg(self) -> Union[Config, None]:
        """Initialize model_cfg for override recipe's model configuration."""
        raise NotImplementedError("this method should be implemented")

    def _init_train_data_cfg(self, dataset: DatasetEntity) -> Union[Config, None]:
        """Initialize data_cfg for override recipe's data configuration."""
        return ConfigDict(data=dataset) if dataset else self._data_cfg

    def _init_test_data_cfg(self, dataset: DatasetEntity) -> Union[Config, None]:
        """Initialize data_cfg for override recipe's data configuration."""
        return ConfigDict(data=dataset) if dataset else self._data_cfg

    def _init_recipe_hparam(self) -> dict:
        """Initialize recipe hyperparamter as dict."""
        assert self._recipe_cfg is not None

        params = self._hyperparams.learning_parameters
        warmup_iters = int(params.learning_rate_warmup_iters)
        lr_config = (
            ConfigDict(warmup_iters=warmup_iters)
            if warmup_iters > 0
            else ConfigDict(warmup_iters=warmup_iters, warmup=None)
        )

        if params.enable_early_stopping and self._recipe_cfg.get("evaluation", None):
            early_stop = ConfigDict(
                start=int(params.early_stop_start),
                patience=int(params.early_stop_patience),
                iteration_patience=int(params.early_stop_iteration_patience),
            )
        else:
            early_stop = False

        runner = ConfigDict(max_epochs=int(params.num_iters))
        if self._recipe_cfg.get("runner", None) and self._recipe_cfg.runner.get("type").startswith("IterBasedRunner"):
            runner = ConfigDict(max_iters=int(params.num_iters))

        return ConfigDict(
            optimizer=ConfigDict(lr=params.learning_rate),
            lr_config=lr_config,
            early_stop=early_stop,
            data=ConfigDict(
                samples_per_gpu=int(params.batch_size),
                workers_per_gpu=int(params.num_workers),
            ),
            runner=runner,
        )

    def _update_stage_module(self, stage_module: str):
        return stage_module

    def _init_deploy_cfg(self) -> Union[Config, None]:
        base_dir = os.path.abspath(os.path.dirname(self.template_file_path))
        deploy_cfg_path = os.path.join(base_dir, "deployment.py")
        deploy_cfg = None
        if os.path.exists(deploy_cfg_path):
            deploy_cfg = MPAConfig.fromfile(deploy_cfg_path)

            def patch_input_preprocessing(deploy_cfg):
                normalize_cfg = get_configs_by_pairs(
                    self._recipe_cfg.data.test.pipeline,
                    dict(type="Normalize"),
                )
                assert len(normalize_cfg) == 1
                normalize_cfg = normalize_cfg[0]

                options = dict(flags=[], args={})
                # NOTE: OTX loads image in RGB format
                # so that `to_rgb=True` means a format change to BGR instead.
                # Conventionally, OpenVINO IR expects a image in BGR format
                # but OpenVINO IR under OTX assumes a image in RGB format.
                #
                # `to_rgb=True` -> a model was trained with images in BGR format
                #                  and a OpenVINO IR needs to reverse input format from RGB to BGR
                # `to_rgb=False` -> a model was trained with images in RGB format
                #                   and a OpenVINO IR does not need to do a reverse
                if normalize_cfg.get("to_rgb", False):
                    options["flags"] += ["--reverse_input_channels"]
                # value must be a list not a tuple
                if normalize_cfg.get("mean", None) is not None:
                    options["args"]["--mean_values"] = list(normalize_cfg.get("mean"))
                if normalize_cfg.get("std", None) is not None:
                    options["args"]["--scale_values"] = list(normalize_cfg.get("std"))

                # fill default
                backend_config = deploy_cfg.backend_config
                if backend_config.get("mo_options") is None:
                    backend_config.mo_options = ConfigDict()
                mo_options = backend_config.mo_options
                if mo_options.get("args") is None:
                    mo_options.args = ConfigDict()
                if mo_options.get("flags") is None:
                    mo_options.flags = []

                # already defiend options have higher priority
                options["args"].update(mo_options.args)
                mo_options.args = ConfigDict(options["args"])
                # make sure no duplicates
                mo_options.flags.extend(options["flags"])
                mo_options.flags = list(set(mo_options.flags))

            def patch_input_shape(deploy_cfg):
                resize_cfg = get_configs_by_pairs(
                    self._recipe_cfg.data.test.pipeline,
                    dict(type="Resize"),
                )
                assert len(resize_cfg) == 1
                resize_cfg = resize_cfg[0]
                size = resize_cfg.size
                if isinstance(size, int):
                    size = (size, size)
                assert all(isinstance(i, int) and i > 0 for i in size)
                # default is static shape to prevent an unexpected error
                # when converting to OpenVINO IR
                deploy_cfg.backend_config.model_inputs = [ConfigDict(opt_shapes=ConfigDict(input=[1, 3, *size]))]

            patch_input_preprocessing(deploy_cfg)
            if not deploy_cfg.backend_config.get("model_inputs", []):
                patch_input_shape(deploy_cfg)

        return deploy_cfg

    def _load_model_ckpt(self, model: Optional[ModelEntity]):
        if model and "weights.pth" in model.model_adapters:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))

            # set confidence_threshold as well
            self.confidence_threshold = model_data.get("confidence_threshold", self.confidence_threshold)
            if model_data.get("anchors"):
                self._anchors = model_data["anchors"]

            # Get config
            if model_data.get("config"):
                tiling_parameters = model_data.get("config").get("tiling_parameters")
                if tiling_parameters and tiling_parameters["enable_tiling"]["value"]:
                    logger.info("Load tiling parameters")
                    self._hyperparams.tiling_parameters.enable_tiling = tiling_parameters["enable_tiling"]["value"]
                    self._hyperparams.tiling_parameters.tile_size = tiling_parameters["tile_size"]["value"]
                    self._hyperparams.tiling_parameters.tile_overlap = tiling_parameters["tile_overlap"]["value"]
                    self._hyperparams.tiling_parameters.tile_max_number = tiling_parameters["tile_max_number"]["value"]
            return model_data
        return None

    def _load_model_label_schema(self, model: Optional[ModelEntity]):
        # If a model has been trained and saved for the task already, create empty model and load weights here
        if model and "label_schema.json" in model.model_adapters:
            import json

            buffer = json.loads(model.get_data("label_schema.json").decode("utf-8"))
            model_label_schema = LabelSchemaMapper().backward(buffer)
            return model_label_schema.get_labels(include_empty=False)
        return self._labels

    def _load_resume_info(self, model: Optional[ModelEntity]):
        if model and "resume" in model.model_adapters:
            return model.model_adapters.get("resume", False)
        return False

    @staticmethod
    def _get_confidence_threshold(hyperparams):
        confidence_threshold = 0.3
        if hasattr(hyperparams, "postprocessing") and hasattr(hyperparams.postprocessing, "confidence_threshold"):
            confidence_threshold = hyperparams.postprocessing.confidence_threshold
        return confidence_threshold

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

    def cancel_hook_initialized(self, cancel_interface: CancelInterfaceHook):
        """Initialization of cancel_interface hook."""
        logger.info("cancel hook is initialized")
        self.cancel_interface = cancel_interface
        if self.reserved_cancel and self.cancel_interface:
            self.cancel_interface.cancel()

    def unload(self):
        """Unload the task."""
        if self._work_dir_is_temp:
            self._delete_scratch_space()
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

    def cleanup(self):
        """Clean up work directory if user specified it."""
        if self._work_dir_is_temp:
            self._delete_scratch_space()

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
            result.__findable = True  # pylint: disable=unused-private-member
            return result

        def __reduce__(self):
            """Function reduce in OnHookInitialized."""
            return (self.__class__, (id(self.task_instance),))

    def update_override_configurations(self, config):
        """Update override_configs."""
        logger.info(f"update override config with: {config}")
        config = ConfigDict(**config)
        self.override_configs.update(config)

    def set_early_stopping_hook(self):
        """Update Early-stopping Hook."""
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

    def _update_caching_modules(self, data_cfg: Config) -> None:
        def _find_max_num_workers(cfg: dict):
            num_workers = [0]
            for key, value in cfg.items():
                if key == "workers_per_gpu" and isinstance(value, int):
                    num_workers += [value]
                elif isinstance(value, dict):
                    num_workers += [_find_max_num_workers(value)]

            return max(num_workers)

        def _get_mem_cache_size():
            if not hasattr(self.hyperparams.algo_backend, "mem_cache_size"):
                return 0

            return self.hyperparams.algo_backend.mem_cache_size

        max_num_workers = _find_max_num_workers(data_cfg)
        mem_cache_size = _get_mem_cache_size()

        mode = "multiprocessing" if max_num_workers > 0 else "singleprocessing"
        caching.MemCacheHandlerSingleton.create(mode, mem_cache_size)

        update_or_add_custom_hook(
            self._recipe_cfg,
            ConfigDict(type="MemCacheHook", priority="VERY_LOW"),
        )
