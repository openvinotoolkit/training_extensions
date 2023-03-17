"""Task of OTX Detection using mmdetection training backend."""

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
import os
import shutil
import tempfile
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Union

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from mmcv.utils import Config, ConfigDict

# This should be removed
# pylint: disable=unused-import
import otx.mpa.det  # noqa
from otx.algorithms.common.adapters.mmcv.hooks import OTXLoggerHook
from otx.algorithms.common.adapters.mmcv.utils import (
    align_data_config_with_recipe,
    get_configs_by_keys,
    get_configs_by_pairs,
    patch_data_pipeline,
    patch_default_config,
    patch_runner,
)
from otx.algorithms.common.configs.training_base import TrainType
from otx.algorithms.common.utils import UncopiableDefaultDict
from otx.algorithms.common.utils.callback import (
    InferenceProgressCallback,
    TrainingProgressCallback,
)
from otx.algorithms.common.utils.data import get_dataset
from otx.algorithms.common.utils.ir import embed_ir_model_data
from otx.algorithms.detection.adapters.mmdet.utils import (
    patch_datasets,
    patch_evaluation,
)
from otx.algorithms.detection.adapters.mmdet.utils.builder import build_detector
from otx.algorithms.detection.adapters.mmdet.utils.config_utils import (
    cluster_anchors,
    should_cluster_anchors,
)
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.utils import get_det_model_api_configuration
from otx.algorithms.detection.utils.data import adaptive_tile_params
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import config_to_bytes, ids_to_strings
from otx.api.entities.annotation import Annotation
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.metrics import (
    BarChartInfo,
    BarMetricsGroup,
    CurveMetric,
    LineChartInfo,
    LineMetricsGroup,
    MetricsGroup,
    ScoreMetric,
    VisualizationType,
)
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.model_template import TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.entities.train_parameters import TrainParameters, default_progress_callback
from otx.api.serialization.label_mapper import LabelSchemaMapper, label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.utils.dataset_utils import add_saliency_maps_to_dataset_item
from otx.core.data import caching
from otx.mpa.builder import build
from otx.mpa.modules.hooks.cancel_interface_hook import CancelInterfaceHook
from otx.mpa.stage import Stage
from otx.mpa.utils.config_utils import (
    MPAConfig,
    add_custom_hook_if_not_exists,
    remove_custom_hook,
    update_or_add_custom_hook,
)
from otx.mpa.utils.logger import get_logger

logger = get_logger()

RECIPE_TRAIN_TYPE = {
    TrainType.SEMISUPERVISED: "semisl.py",
    TrainType.INCREMENTAL: "incremental.py",
}

TRAIN_TYPE_DIR_PATH = {
    TrainType.INCREMENTAL.name: ".",
    TrainType.SELFSUPERVISED.name: "selfsl",
    TrainType.SEMISUPERVISED.name: "semisl",
}

# TODO Remove unnecessary pylint disable
# pylint: disable=too-many-lines


# This should be moved to OTX side
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


class MMDetectionTask:
    """Task class for OTX detection using mmdetection training backend."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        self._task_config = DetectionConfig
        self._task_environment = task_environment
        self._task_type = task_environment.model_template.task_type
        self._labels = task_environment.get_labels(include_empty=False)
        self._hyperparams = task_environment.get_hyper_parameters(self._task_config)  # type: ConfigDict
        self._train_type = self._hyperparams.algo_backend.train_type
        self._model_dir = os.path.join(
            os.path.abspath(os.path.dirname(self._task_environment.model_template.model_template_path)),
            TRAIN_TYPE_DIR_PATH[self._train_type.name],
        )
        if self._hyperparams.tiling_parameters.enable_tiling:
            self.data_pipeline_path = os.path.join(self._model_dir, "tile_pipeline.py")
        else:
            self.data_pipeline_path = os.path.join(self._model_dir, "data_pipeline.py")
        self._work_dir_is_temp = False
        self._output_path = output_path
        if self._output_path is None:
            self._output_path = tempfile.mkdtemp(prefix="OTX-task-")
            self._work_dir_is_temp = True
        self._anchors: Dict[str, int] = {}
        self._time_monitor = None
        self.on_hook_initialized = OnHookInitialized(self)
        self._learning_curves = UncopiableDefaultDict(OTXLoggerHook.Curve)
        self._model_label_schema = []  # type: List[LabelEntity]
        self._resume = False
        self._should_stop = False
        self.cancel_interface = None
        self.reserved_cancel = False
        self._model_ckpt = None
        self._precision = [ModelPrecision.FP32]
        self._data_cfg = None
        self._optimization_methods = []  # type: List[OptimizationMethod]
        self._recipe_cfg: Optional[Config] = None
        self._is_training = False
        self._mode = ""

        if hasattr(self._hyperparams, "postprocessing") and hasattr(
            self._hyperparams.postprocessing, "confidence_threshold"
        ):
            self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold
        else:
            self.confidence_threshold = 0.0

        # This should be removed
        self.override_configs = {}  # type: Dict[str, str]

        # This should be cleaned
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

        # This is for hpo, and this should be removed
        self.project_path = self._output_path

    # This funciton should be moved to load_model
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

    # This function should be moved to load_model
    def _load_model_label_schema(self, model: Optional[ModelEntity]):
        # If a model has been trained and saved for the task already, create empty model and load weights here
        if model and "label_schema.json" in model.model_adapters:
            import json

            buffer = json.loads(model.get_data("label_schema.json").decode("utf-8"))
            model_label_schema = LabelSchemaMapper().backward(buffer)
            return model_label_schema.get_labels(include_empty=False)
        return self._labels

    # This function should be moved to load_model
    def _load_resume_info(self, model: Optional[ModelEntity]):
        if model and "resume" in model.model_adapters:
            return model.model_adapters.get("resume", False)
        return False

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _init_task(self, export: bool = False):  # noqa
        """Initialize task."""
        self._recipe_cfg = MPAConfig.fromfile(os.path.join(self._model_dir, "model.py"))
        if len(self._anchors) != 0:
            self._update_anchors(self._recipe_cfg.model.bbox_head.anchor_generator, self._anchors)

        # This may go to the configure function
        options_for_patch_datasets = {"type": "OTXDetDataset"}
        patch_default_config(self._recipe_cfg)
        patch_runner(self._recipe_cfg)
        patch_data_pipeline(self._recipe_cfg, self.data_pipeline_path)
        patch_datasets(
            self._recipe_cfg,
            self._task_type.domain,
            **options_for_patch_datasets,
        )  # for OTX compatibility
        patch_evaluation(self._recipe_cfg)  # for OTX compatibility

        # This may go to the configure function
        if not export:
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
            if self._recipe_cfg.get("runner", None) and self._recipe_cfg.runner.get("type").startswith(
                "IterBasedRunner"
            ):
                runner = ConfigDict(max_iters=int(params.num_iters))

            hparams = ConfigDict(
                optimizer=ConfigDict(lr=params.learning_rate),
                lr_config=lr_config,
                early_stop=early_stop,
                data=ConfigDict(
                    samples_per_gpu=int(params.batch_size),
                    workers_per_gpu=int(params.num_workers),
                ),
                runner=runner,
            )
            if bool(self._hyperparams.tiling_parameters.enable_tiling):
                logger.info("Tiling Enabled")
                tiling_params = ConfigDict(
                    tile_size=int(self._hyperparams.tiling_parameters.tile_size),
                    overlap_ratio=float(self._hyperparams.tiling_parameters.tile_overlap),
                    max_per_img=int(self._hyperparams.tiling_parameters.tile_max_number),
                )
                hparams.update(
                    ConfigDict(
                        data=ConfigDict(
                            train=tiling_params,
                            val=tiling_params,
                            test=tiling_params,
                        )
                    )
                )
                hparams.update(dict(evaluation=dict(iou_thr=[0.5])))

            hparams["use_adaptive_interval"] = self._hyperparams.learning_parameters.use_adaptive_interval
            self._recipe_cfg.merge_from_dict(hparams)

        # This may go to configure function
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
                del self._recipe_cfg._cfg_dict["fp16"]  # pylint: disable=protected-access
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

            # if self._anchors are set somewhere, anchors had already been clusted
            # by this method or by loading trained model
            if should_cluster_anchors(self._recipe_cfg) and len(self._anchors) == 0:
                otx_dataset = get_configs_by_keys(self._data_cfg.data.train, "otx_dataset")
                assert len(otx_dataset) == 1
                otx_dataset = otx_dataset[0]
                cluster_anchors(
                    self._recipe_cfg,
                    otx_dataset,
                )
                self._update_anchors(self._anchors, self._recipe_cfg.model.bbox_head.anchor_generator)

        logger.info("initialized.")

    @staticmethod
    def _update_anchors(origin, new):
        logger.info("Updating anchors")
        origin["heights"] = new["heights"]
        origin["widths"] = new["widths"]

    def configure(self):
        """Patch mmcv configs for OTX detection settings."""
        return

    # pylint: disable=too-many-branches, too-many-statements
    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: Optional[TrainParameters] = None,
    ):
        """Train function in DetectionTrainTask."""
        logger.info("train()")
        # Check for stop signal when training has stopped.
        # If should_stop is true, training was cancelled and no new
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return

        # Set OTE LoggerHook & Time Monitor
        if train_parameters:
            update_progress_callback = train_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback
        self._time_monitor = TrainingProgressCallback(update_progress_callback)

        logger.info("init data cfg.")
        self._data_cfg = ConfigDict(data=ConfigDict())

        for cfg_key, subset in zip(
            ["train", "val", "unlabeled"],
            [Subset.TRAINING, Subset.VALIDATION, Subset.UNLABELED],
        ):
            subset = get_dataset(dataset, subset)
            if subset and self._data_cfg is not None:
                self._data_cfg.data[cfg_key] = ConfigDict(
                    otx_dataset=subset,
                    labels=self._labels,
                )

        # Temparory remedy for cfg.pretty_text error
        for label in self._labels:
            label.hotkey = "a"

        self._is_training = True

        if bool(self._hyperparams.tiling_parameters.enable_tiling) and bool(
            self._hyperparams.tiling_parameters.enable_adaptive_params
        ):
            adaptive_tile_params(self._hyperparams.tiling_parameters, dataset)

        self._init_task()

        stage_module = "DetectionTrainer"
        module_prefix = {TrainType.INCREMENTAL: "Incr", TrainType.SEMISUPERVISED: "SemiSL"}
        stage_module = module_prefix[self._train_type] + stage_module

        mode = "train"
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

        common_cfg = ConfigDict(dict(output_path=self._output_path, resume=self._resume))

        # build workflow using recipe configuration
        workflow = build(
            recipe_cfg,
            self._mode,
            stage_type=stage_module,
            common_cfg=common_cfg,
        )

        # run workflow with task specific model config and data config
        results = workflow.run(
            model_cfg=recipe_cfg,
            data_cfg=data_cfg,
            ir_model_path=None,
            ir_weight_path=None,
            model_ckpt=self._model_ckpt,
            mode=self._mode,
        )
        logger.info("run task done.")

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return

        # get output model
        model_ckpt = results.get("final_ckpt")
        if model_ckpt is None:
            logger.error("cannot find final checkpoint from the results.")
            # output_model.model_status = ModelStatus.FAILED
            return
        # update checkpoint to the newly trained model
        self._model_ckpt = model_ckpt

        # get prediction on validation set
        self._is_training = False
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        val_preds, val_map = self.infer(
            val_dataset, InferenceParameters(is_evaluation=True), init_task=False, add_prediction=False
        )

        preds_val_dataset = val_dataset.with_empty_annotations()
        self._add_predictions_to_dataset(val_preds, preds_val_dataset, 0.0)

        result_set = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=val_dataset,
            prediction_dataset=preds_val_dataset,
        )

        # adjust confidence threshold
        if self._hyperparams.postprocessing.result_based_confidence_threshold:
            best_confidence_threshold = None
            logger.info("Adjusting the confidence threshold")
            metric = MetricsHelper.compute_f_measure(result_set, vary_confidence_threshold=True)
            if metric.best_confidence_threshold:
                best_confidence_threshold = metric.best_confidence_threshold.value
            if best_confidence_threshold is None:
                raise ValueError("Cannot compute metrics: Invalid confidence threshold!")
            logger.info(f"Setting confidence threshold to {best_confidence_threshold} based on results")
            self.confidence_threshold = best_confidence_threshold
        else:
            metric = MetricsHelper.compute_f_measure(result_set, vary_confidence_threshold=False)

        # compose performance statistics
        # TODO[EUGENE]: HOW TO ADD A MAE CURVE FOR TaskType.COUNTING?
        performance = metric.get_performance()
        performance.dashboard_metrics.extend(self._generate_training_metrics(self._learning_curves, val_map))
        logger.info(f"Final model performance: {str(performance)}")
        # save resulting model
        self.save_model(output_model)
        output_model.performance = performance
        logger.info("train done.")

    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
        init_task: bool = True,
        add_prediction: bool = True,
    ) -> DatasetEntity:
        """Main infer function."""
        logger.info("infer()")
        process_saliency_maps = False
        explain_predicted_classes = True

        if init_task:
            update_progress_callback = default_progress_callback
            if inference_parameters is not None:
                update_progress_callback = inference_parameters.update_progress  # type: ignore
                process_saliency_maps = inference_parameters.process_saliency_maps
                explain_predicted_classes = inference_parameters.explain_predicted_classes

            self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)
            # If confidence threshold is adaptive then up-to-date value should be stored in the model
            # and should not be changed during inference. Otherwise user-specified value should be taken.
            if not self._hyperparams.postprocessing.result_based_confidence_threshold:
                self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold
            logger.info(f"Confidence threshold {self.confidence_threshold}")

        self._data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    otx_dataset=None,
                    labels=self._labels,
                ),
                test=ConfigDict(
                    otx_dataset=dataset,
                    labels=self._labels,
                ),
            )
        )

        dump_features = True
        dump_saliency_map = not inference_parameters.is_evaluation if inference_parameters else True

        self._init_task()

        stage_module = "DetectionInferrer"
        module_prefix = {TrainType.INCREMENTAL: "Incr", TrainType.SEMISUPERVISED: "SemiSL"}
        stage_module = module_prefix[self._train_type] + stage_module

        mode = "train"
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

        common_cfg = ConfigDict(dict(output_path=self._output_path, resume=self._resume))

        # build workflow using recipe configuration
        workflow = build(
            recipe_cfg,
            self._mode,
            stage_type=stage_module,
            common_cfg=common_cfg,
        )

        # run workflow with task specific model config and data config
        results = workflow.run(
            model_cfg=recipe_cfg,
            data_cfg=data_cfg,
            ir_model_path=None,
            ir_weight_path=None,
            model_ckpt=self._model_ckpt,
            mode=self._mode,
            eval=inference_parameters.is_evaluation if inference_parameters else False,
            dump_features=dump_features,
            dump_saliency_map=dump_saliency_map,
        )
        logger.info("run task done.")
        # TODO: InferenceProgressCallback register
        logger.debug(f"result of run_task {stage_module} module = {results}")
        output = results["outputs"]
        metric = output["metric"]
        predictions = output["detections"]
        assert len(output["detections"]) == len(output["feature_vectors"]) == len(output["saliency_maps"]), (
            "Number of elements should be the same, however, number of outputs are "
            f"{len(output['detections'])}, {len(output['feature_vectors'])}, and {len(output['saliency_maps'])}"
        )
        prediction_results = zip(predictions, output["feature_vectors"], output["saliency_maps"])

        if add_prediction:
            self._add_predictions_to_dataset(
                prediction_results, dataset, self.confidence_threshold, process_saliency_maps, explain_predicted_classes
            )
            logger.info("Inference completed")
            return dataset
        return prediction_results, metric

    # pylint: disable=too-many-statements
    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = True,
    ):
        """Export function of OTX Detection Task."""
        # copied from OTX inference_task.py
        logger.info("Exporting the model")
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f"not supported export type {export_type}")
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        self._init_task(export=True)

        if precision == ModelPrecision.FP16:
            self._precision[0] = ModelPrecision.FP16
        export_options: Dict[str, Any] = {}
        export_options["deploy_cfg"] = self._init_deploy_cfg()
        if export_options.get("precision", None) is None:
            assert len(self._precision) == 1
            export_options["precision"] = str(self._precision[0])

        export_options["deploy_cfg"]["dump_features"] = dump_features
        if dump_features:
            output_names = export_options["deploy_cfg"]["ir_config"]["output_names"]
            if "feature_vector" not in output_names and "saliency_map" not in output_names:
                export_options["deploy_cfg"]["ir_config"]["output_names"] += ["feature_vector", "saliency_map"]
        export_options["model_builder"] = build_detector

        stage_module = "DetectionExporter"

        mode = "train"
        if mode is not None:
            self._mode = mode

        # deepcopy all configs to make sure
        # changes under MPA and below does not take an effect to OTX for clear distinction
        recipe_cfg = deepcopy(self._recipe_cfg)
        data_cfg = deepcopy(self._data_cfg)
        assert recipe_cfg is not None, "'recipe_cfg' is not initialized."

        # update model config -> model label schema
        model_classes = [label.name for label in self._model_label_schema]
        recipe_cfg["model_classes"] = model_classes
        common_cfg = ConfigDict(dict(output_path=self._output_path, resume=self._resume))

        # build workflow using recipe configuration
        workflow = build(
            recipe_cfg,
            self._mode,
            stage_type=stage_module,
            common_cfg=common_cfg,
        )

        # run workflow with task specific model config and data config
        results = workflow.run(
            model_cfg=recipe_cfg,
            data_cfg=data_cfg,
            ir_model_path=None,
            ir_weight_path=None,
            model_ckpt=self._model_ckpt,
            mode=self._mode,
            export=True,
            dump_features=dump_features,
            enable_fp16=(precision == ModelPrecision.FP16),
            **export_options,
        )
        outputs = results.get("outputs")
        logger.debug(f"results of run_task = {outputs}")
        if outputs is None:
            raise RuntimeError(results.get("msg"))

        bin_file = outputs.get("bin")
        xml_file = outputs.get("xml")

        ir_extra_data = get_det_model_api_configuration(
            self._task_environment.label_schema, self._task_type, self.confidence_threshold
        )
        embed_ir_model_data(xml_file, ir_extra_data)

        if xml_file is None or bin_file is None:
            raise RuntimeError("invalid status of exporting. bin and xml should not be None")
        with open(bin_file, "rb") as f:
            output_model.set_data("openvino.bin", f.read())
        with open(xml_file, "rb") as f:
            output_model.set_data("openvino.xml", f.read())
        output_model.set_data(
            "confidence_threshold",
            np.array([self.confidence_threshold], dtype=np.float32).tobytes(),
        )
        output_model.set_data("config.json", config_to_bytes(self._hyperparams))
        output_model.precision = self._precision
        output_model.optimization_methods = self._optimization_methods
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        logger.info("Exporting completed")

    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Main explain function of OTX Detection."""
        logger.info("explain()")

        update_progress_callback = default_progress_callback
        process_saliency_maps = False
        explain_predicted_classes = True
        if explain_parameters is not None:
            update_progress_callback = explain_parameters.update_progress  # type: ignore
            process_saliency_maps = explain_parameters.process_saliency_maps
            explain_predicted_classes = explain_parameters.explain_predicted_classes

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

        self._init_task()

        stage_module = "DetectionExplainer"
        self._data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    otx_dataset=None,
                    labels=self._labels,
                ),
                test=ConfigDict(
                    otx_dataset=dataset,
                    labels=self._labels,
                ),
            )
        )
        mode = "train"
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

        common_cfg = ConfigDict(dict(output_path=self._output_path, resume=self._resume))

        # build workflow using recipe configuration
        workflow = build(
            recipe_cfg,
            self._mode,
            stage_type=stage_module,
            common_cfg=common_cfg,
        )

        # run workflow with task specific model config and data config
        results = workflow.run(
            model_cfg=recipe_cfg,
            data_cfg=data_cfg,
            ir_model_path=None,
            ir_weight_path=None,
            model_ckpt=self._model_ckpt,
            mode=self._mode,
            explainer=explain_parameters.explainer if explain_parameters else None,
        )
        logger.info("run task done.")
        detections = results["outputs"]["detections"]
        explain_results = results["outputs"]["saliency_maps"]

        self._add_explanations_to_dataset(
            detections, explain_results, dataset, process_saliency_maps, explain_predicted_classes
        )
        logger.info("Explain completed")
        return dataset

    # This should be moved to OTXDetectionDataset
    def evaluate(
        self,
        output_resultset: ResultSetEntity,
        evaluation_metric: Optional[str] = None,
    ):
        """Evaluate function of OTX Detection Task."""
        logger.info("called evaluate()")
        if evaluation_metric is not None:
            logger.warning(
                f"Requested to use {evaluation_metric} metric, " "but parameter is ignored. Use F-measure instead."
            )
        metric = MetricsHelper.compute_f_measure(output_resultset)
        logger.info(f"F-measure after evaluation: {metric.f_measure.value}")
        output_resultset.performance = metric.get_performance()
        logger.info("Evaluation completed")

    # This should be removed
    def update_override_configurations(self, config):
        """Update override_configs."""
        logger.info(f"update override config with: {config}")
        config = ConfigDict(**config)
        self.override_configs.update(config)

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
            if not hasattr(self._hyperparams.algo_backend, "mem_cache_size"):
                return 0

            return self._hyperparams.algo_backend.mem_cache_size

        max_num_workers = _find_max_num_workers(data_cfg)
        mem_cache_size = _get_mem_cache_size()

        mode = "multiprocessing" if max_num_workers > 0 else "singleprocessing"
        caching.MemCacheHandlerSingleton.create(mode, mem_cache_size)

        update_or_add_custom_hook(
            self._recipe_cfg,
            ConfigDict(type="MemCacheHook", priority="VERY_LOW"),
        )

    # This should moved somewhere
    def _init_deploy_cfg(self) -> Union[Config, None]:
        base_dir = os.path.abspath(os.path.dirname(self._task_environment.model_template.model_template_path))
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

    # Below functions should be moved to OTXDetectionTask
    def _add_predictions_to_dataset(
        self,
        prediction_results,
        dataset,
        confidence_threshold=0.0,
        process_saliency_maps=False,
        explain_predicted_classes=True,
    ):
        """Loop over dataset again to assign predictions. Convert from MMDetection format to OTX format."""
        for dataset_item, (all_results, feature_vector, saliency_map) in zip(dataset, prediction_results):
            shapes = self._get_shapes(all_results, dataset_item.width, dataset_item.height, confidence_threshold)
            dataset_item.append_annotations(shapes)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)

            if saliency_map is not None:
                labels = self._labels.copy()
                if saliency_map.shape[0] == len(labels) + 1:
                    # Include the background as the last category
                    labels.append(LabelEntity("background", Domain.DETECTION))

                predicted_scored_labels = []
                for shape in shapes:
                    predicted_scored_labels += shape.get_labels()

                add_saliency_maps_to_dataset_item(
                    dataset_item=dataset_item,
                    saliency_map=saliency_map,
                    model=self._task_environment.model,
                    labels=labels,
                    predicted_scored_labels=predicted_scored_labels,
                    explain_predicted_classes=explain_predicted_classes,
                    process_saliency_maps=process_saliency_maps,
                )

    def _get_shapes(self, all_results, width, height, confidence_threshold):
        if self._task_type == TaskType.DETECTION:
            shapes = self._det_add_predictions_to_dataset(all_results, width, height, confidence_threshold)
        elif self._task_type in {
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.ROTATED_DETECTION,
        }:
            shapes = self._ins_seg_add_predictions_to_dataset(all_results, width, height, confidence_threshold)
        else:
            raise RuntimeError(f"MPA results assignment not implemented for task: {self._task_type}")
        return shapes

    def _det_add_predictions_to_dataset(self, all_results, width, height, confidence_threshold):
        shapes = []
        for label_idx, detections in enumerate(all_results):
            for i in range(detections.shape[0]):
                probability = float(detections[i, 4])
                coords = detections[i, :4].astype(float).copy()
                coords /= np.array([width, height, width, height], dtype=float)
                coords = np.clip(coords, 0, 1)

                if probability < confidence_threshold:
                    continue

                assigned_label = [ScoredLabel(self._labels[label_idx], probability=probability)]
                if coords[3] - coords[1] <= 0 or coords[2] - coords[0] <= 0:
                    continue

                shapes.append(
                    Annotation(
                        Rectangle(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                        labels=assigned_label,
                    )
                )
        return shapes

    def _ins_seg_add_predictions_to_dataset(self, all_results, width, height, confidence_threshold):
        shapes = []
        for label_idx, (boxes, masks) in enumerate(zip(*all_results)):
            for mask, probability in zip(masks, boxes[:, 4]):
                if isinstance(mask, dict):
                    mask = mask_util.decode(mask)
                mask = mask.astype(np.uint8)
                probability = float(probability)
                contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hierarchies is None:
                    continue
                for contour, hierarchy in zip(contours, hierarchies[0]):
                    if hierarchy[3] != -1:
                        continue
                    if len(contour) <= 2 or probability < confidence_threshold:
                        continue
                    if self._task_type == TaskType.INSTANCE_SEGMENTATION:
                        points = [Point(x=point[0][0] / width, y=point[0][1] / height) for point in contour]
                    else:
                        box_points = cv2.boxPoints(cv2.minAreaRect(contour))
                        points = [Point(x=point[0] / width, y=point[1] / height) for point in box_points]
                    labels = [ScoredLabel(self._labels[label_idx], probability=probability)]
                    polygon = Polygon(points=points)
                    if cv2.contourArea(contour) > 0 and polygon.get_area() > 1e-12:
                        shapes.append(Annotation(polygon, labels=labels, id=ID(f"{label_idx:08}")))
        return shapes

    def _add_explanations_to_dataset(
        self, detections, explain_results, dataset, process_saliency_maps, explain_predicted_classes
    ):
        """Add saliency map to the dataset."""
        for dataset_item, detection, saliency_map in zip(dataset, detections, explain_results):
            labels = self._labels.copy()
            if saliency_map.shape[0] == len(labels) + 1:
                # Include the background as the last category
                labels.append(LabelEntity("background", Domain.DETECTION))

            shapes = self._get_shapes(detection, dataset_item.width, dataset_item.height, 0.4)
            predicted_scored_labels = []
            for shape in shapes:
                predicted_scored_labels += shape.get_labels()

            add_saliency_maps_to_dataset_item(
                dataset_item=dataset_item,
                saliency_map=saliency_map,
                model=self._task_environment.model,
                labels=labels,
                predicted_scored_labels=predicted_scored_labels,
                explain_predicted_classes=explain_predicted_classes,
                process_saliency_maps=process_saliency_maps,
            )

    @staticmethod
    def _generate_training_metrics(learning_curves, scores) -> Iterable[MetricsGroup[Any, Any]]:
        """Get Training metrics (epochs & scores).

        Parses the mmdetection logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        # Learning curves.
        for key, curve in learning_curves.items():
            len_x, len_y = len(curve.x), len(curve.y)
            if len_x != len_y:
                logger.warning(f"Learning curve {key} has inconsistent number of coordinates ({len_x} vs {len_y}.")
                len_x = min(len_x, len_y)
                curve.x = curve.x[:len_x]
                curve.y = curve.y[:len_x]
            metric_curve = CurveMetric(
                xs=np.nan_to_num(curve.x).tolist(),
                ys=np.nan_to_num(curve.y).tolist(),
                name=key,
            )
            visualization_info = LineChartInfo(name=key, x_axis_label="Epoch", y_axis_label=key)
            output.append(LineMetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        # Final mAP value on the validation set.
        output.append(
            BarMetricsGroup(
                metrics=[ScoreMetric(value=scores, name="mAP")],
                visualization_info=BarChartInfo("Validation score", visualization_type=VisualizationType.RADIAL_BAR),
            )
        )

        return output

    def save_model(self, output_model: ModelEntity):
        """Save best model weights in DetectionTrainTask."""
        logger.info("called save_model")
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {
            "model": model_ckpt,
            "config": hyperparams_str,
            "labels": labels,
            "confidence_threshold": self.confidence_threshold,
            "VERSION": 1,
        }
        if self._recipe_cfg is not None and should_cluster_anchors(self._recipe_cfg):
            modelinfo["anchors"] = {}
            self._update_anchors(modelinfo["anchors"], self._recipe_cfg.model.bbox_head.anchor_generator)

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        output_model.precision = self._precision

    def cancel_training(self):
        """Cancel training function in DetectionTrainTask.

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
        """Remove model checkpoints and mpa logs."""
        if os.path.exists(self._output_path):
            shutil.rmtree(self._output_path, ignore_errors=False)
