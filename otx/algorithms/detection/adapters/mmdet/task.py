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

import glob
import io
import os
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from mmcv.runner import wrap_fp16_model
from mmcv.utils import Config, ConfigDict, get_git_hash
from mmdet import __version__
from mmdet.apis import single_gpu_test, train_detector
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models.detectors import TwoStageDetector
from mmdet.utils import collect_env

from otx.algorithms.common.adapters.mmcv.utils import (
    align_data_config_with_recipe,
    build_data_parallel,
    get_configs_by_keys,
    get_configs_by_pairs,
    patch_data_pipeline,
    patch_default_config,
    patch_runner,
)
from otx.algorithms.common.configs.training_base import TrainType
from otx.algorithms.common.utils import set_random_seed
from otx.algorithms.common.utils.callback import InferenceProgressCallback
from otx.algorithms.common.utils.data import get_dataset
from otx.algorithms.common.utils.ir import embed_ir_model_data
from otx.algorithms.detection.adapters.mmdet.datasets import ImageTilingDataset
from otx.algorithms.detection.adapters.mmdet.exporter import DetectionExporter
from otx.algorithms.detection.adapters.mmdet.hooks.det_saliency_map_hook import (
    DetSaliencyMapHook,
)
from otx.algorithms.detection.adapters.mmdet.utils import (
    patch_datasets,
    patch_evaluation,
)
from otx.algorithms.detection.adapters.mmdet.utils.builder import build_detector
from otx.algorithms.detection.adapters.mmdet.utils.config_utils import (
    cluster_anchors,
    should_cluster_anchors,
)
from otx.algorithms.detection.adapters.mmdet.utils.configurer import (
    DetectionStage,
    IncrDetectionStage,
    SemiSLDetectionStage,
)
from otx.algorithms.detection.tasks import OTXDetectionTask
from otx.algorithms.detection.utils import get_det_model_api_configuration
from otx.algorithms.detection.utils.data import adaptive_tile_params
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import config_to_bytes, ids_to_strings
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
)
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.train_parameters import default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.core.data import caching
from otx.mpa.modules.hooks.recording_forward_hooks import (
    ActivationMapHook,
    EigenCamHook,
    FeatureVectorHook,
)
from otx.mpa.stage import Stage
from otx.mpa.utils.config_utils import (
    MPAConfig,
    add_custom_hook_if_not_exists,
    remove_custom_hook,
    update_or_add_custom_hook,
)
from otx.mpa.utils.logger import get_logger

logger = get_logger()

# TODO Remove unnecessary pylint disable
# pylint: disable=too-many-lines


class MMDetectionTask(OTXDetectionTask):
    """Task class for OTX detection using mmdetection training backend."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        super().__init__(task_environment, output_path)
        self._data_cfg: Optional[Config] = None
        self._recipe_cfg: Optional[Config] = None

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _init_task(self, export: bool = False):  # noqa
        """Initialize task."""
        self._recipe_cfg = MPAConfig.fromfile(os.path.join(self._model_dir, "model.py"))
        if len(self._anchors) != 0:
            self._update_anchors(self._recipe_cfg.model.bbox_head.anchor_generator, self._anchors)

        set_random_seed(self._recipe_cfg.get("seed", 5), logger, self._recipe_cfg.get("deterministic", False))

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

    def configure(self, model_cfg, model_ckpt, data_cfg, training=True, subset="train", **kwargs):
        """Patch mmcv configs for OTX detection settings."""
        if self._train_type == TrainType.INCREMENTAL:
            configurer = IncrDetectionStage()
        elif self._train_type == TrainType.SEMISUPERVISED:
            configurer = SemiSLDetectionStage()
        else:
            configurer = DetectionStage()
        cfg = configurer.configure(model_cfg, model_ckpt, data_cfg, training, subset, **kwargs)
        return cfg

    # pylint: disable=too-many-branches, too-many-statements
    def _train_model(
        self,
        dataset: DatasetEntity,
    ):
        """Train function in DetectionTrainTask."""
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

        recipe_cfg.work_dir = self._output_path
        recipe_cfg.resume = self._resume

        cfg = self.configure(recipe_cfg, self._model_ckpt, data_cfg, True, "train")
        logger.info("train!")

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # Environment
        logger.info(f"cfg.gpu_ids = {cfg.gpu_ids}, distributed = {cfg.distributed}")
        env_info_dict = collect_env()
        env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
        dash_line = "-" * 60 + "\n"
        logger.info(f"Environment info:\n{dash_line}{env_info}\n{dash_line}")

        # Data
        datasets = [build_dataset(cfg.data.train)]

        # FIXME: Currently detection do not support multi batch evaluation. This will be fixed
        if "val" in cfg.data:
            cfg.data.val_dataloader["samples_per_gpu"] = 1

        # TODO. This should be moved to configurer
        # TODO. Anchor clustering should be checked
        # if hasattr(cfg, "hparams"):
        #     if cfg.hparams.get("adaptive_anchor", False):
        #         num_ratios = cfg.hparams.get("num_anchor_ratios", 5)
        #         proposal_ratio = extract_anchor_ratio(datasets[0], num_ratios)
        #         self.configure_anchor(cfg, proposal_ratio)

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.get("final", [])
        else:
            target_classes = datasets[0].CLASSES

        # Metadata
        meta = dict()
        meta["env_info"] = env_info
        # meta['config'] = cfg.pretty_text
        meta["seed"] = cfg.seed
        meta["exp_name"] = cfg.work_dir
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                CLASSES=target_classes,
            )
            # if "proposal_ratio" in locals():
            #     cfg.checkpoint_config.meta.update({"anchor_ratio": proposal_ratio})

        # Model
        model = build_detector(cfg)
        if getattr(cfg, "fp16", False):
            wrap_fp16_model(model)
        model.train()
        model.CLASSES = target_classes

        if cfg.distributed:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Save config
        # cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
        # logger.info(f'Config:\n{cfg.pretty_text}')

        validate = bool(cfg.data.get("val", None))
        train_detector(
            model,
            datasets,
            cfg,
            distributed=cfg.distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta,
        )

        # Save outputs
        output_ckpt_path = os.path.join(cfg.work_dir, "latest.pth")
        best_ckpt_path = glob.glob(os.path.join(cfg.work_dir, "best_*.pth"))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        logger.info("run task done.")
        return dict(
            final_ckpt=output_ckpt_path,
        )

    def _infer_model(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ):
        """Main infer function."""
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

        recipe_cfg.work_dir = self._output_path
        recipe_cfg.resume = self._resume

        cfg = self.configure(recipe_cfg, self._model_ckpt, data_cfg, False, "test")
        logger.info("infer!")

        samples_per_gpu = cfg.data.test_dataloader.get("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

        # Data loader
        dataset = build_dataset(cfg.data.test)
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=cfg.data.get("samples_per_gpu", 1),
            workers_per_gpu=cfg.data.get("workers_per_gpu", 0),
            num_gpus=len(cfg.gpu_ids),
            dist=cfg.distributed,
            seed=cfg.get("seed", None),
            shuffle=False,
        )

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.final
            if len(target_classes) < 1:
                raise KeyError(
                    f"target_classes={target_classes} is empty check the metadata from model ckpt or recipe "
                    "configuration"
                )
        else:
            target_classes = dataset.CLASSES

        # Model
        model = build_detector(cfg)
        if getattr(cfg, "fp16", False):
            wrap_fp16_model(model)
        model.CLASSES = target_classes
        model.eval()
        feature_model = model.model_t if self._train_type == TrainType.SEMISUPERVISED else model
        model = build_data_parallel(model, cfg, distributed=False)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        time_monitor = None
        if cfg.get("custom_hooks", None):
            time_monitor = [hook.time_monitor for hook in cfg.custom_hooks if hook.type == "OTXProgressHook"]
            time_monitor = time_monitor[0] if time_monitor else None
        if time_monitor is not None:

            # pylint: disable=unused-argument
            def pre_hook(module, inp):
                time_monitor.on_test_batch_begin(None, None)

            def hook(module, inp, outp):
                time_monitor.on_test_batch_end(None, None)

            model.register_forward_pre_hook(pre_hook)
            model.register_forward_hook(hook)

        # Class-wise Saliency map for Single-Stage Detector, otherwise use class-ignore saliency map.
        if not dump_saliency_map:
            saliency_hook = nullcontext()
        else:
            raw_model = feature_model
            if raw_model.__class__.__name__ == "NNCFNetwork":
                raw_model = raw_model.get_nncf_wrapped_model()
            if isinstance(raw_model, TwoStageDetector):
                saliency_hook = ActivationMapHook(feature_model)
            else:
                saliency_hook = DetSaliencyMapHook(feature_model)

        eval_predictions = []
        with FeatureVectorHook(feature_model) if dump_features else nullcontext() as feature_vector_hook:
            with saliency_hook:
                eval_predictions = single_gpu_test(model, dataloader)
                feature_vectors = feature_vector_hook.records if dump_features else [None] * len(dataset)
                if isinstance(saliency_hook, nullcontext):
                    saliency_maps = [None] * len(dataset)
                else:
                    saliency_maps = saliency_hook.records  # pylint: disable=no-member

        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"]:
            cfg.evaluation.pop(key, None)

        metric = None
        if inference_parameters and inference_parameters.is_evaluation:
            metric = dataset.evaluate(eval_predictions, **cfg.evaluation)
            metric = metric["mAP"] if isinstance(cfg.evaluation.metric, list) else metric[cfg.evaluation.metric]

        # Check and unwrap ImageTilingDataset object from TaskAdaptEvalDataset
        while hasattr(dataset, "dataset") and not isinstance(dataset, ImageTilingDataset):
            dataset = dataset.dataset

        if isinstance(dataset, ImageTilingDataset):
            feature_vectors = [feature_vectors[i] for i in range(dataset.num_samples)]
            saliency_maps = [saliency_maps[i] for i in range(dataset.num_samples)]
            if not dataset.merged_results:
                eval_predictions = dataset.merge(eval_predictions)
            else:
                eval_predictions = dataset.merged_results

        assert len(eval_predictions) == len(feature_vectors) == len(saliency_maps), (
            "Number of elements should be the same, however, number of outputs are "
            f"{len(eval_predictions)}, {len(feature_vectors)}, and {len(saliency_maps)}"
        )

        results = dict(
            outputs=dict(
                classes=target_classes,
                detections=eval_predictions,
                metric=metric,
                feature_vectors=feature_vectors,
                saliency_maps=saliency_maps,
            )
        )

        logger.info("run task done.")
        # TODO: InferenceProgressCallback register
        output = results["outputs"]
        metric = output["metric"]
        predictions = output["detections"]
        assert len(output["detections"]) == len(output["feature_vectors"]) == len(output["saliency_maps"]), (
            "Number of elements should be the same, however, number of outputs are "
            f"{len(output['detections'])}, {len(output['feature_vectors'])}, and {len(output['saliency_maps'])}"
        )
        prediction_results = zip(predictions, output["feature_vectors"], output["saliency_maps"])
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

        # deepcopy all configs to make sure
        # changes under MPA and below does not take an effect to OTX for clear distinction
        recipe_cfg = deepcopy(self._recipe_cfg)
        data_cfg = deepcopy(self._data_cfg)
        assert recipe_cfg is not None, "'recipe_cfg' is not initialized."

        # update model config -> model label schema
        model_classes = [label.name for label in self._model_label_schema]
        recipe_cfg["model_classes"] = model_classes

        recipe_cfg.work_dir = self._output_path
        recipe_cfg.resume = self._resume

        cfg = self.configure(recipe_cfg, self._model_ckpt, data_cfg, False, "test")

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

        exporter = DetectionExporter()
        results = exporter.run(
            cfg,
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

        explainer_hook_selector = {
            "classwisesaliencymap": DetSaliencyMapHook,
            "eigencam": EigenCamHook,
            "activationmap": ActivationMapHook,
        }
        logger.info("explain()")

        update_progress_callback = default_progress_callback
        process_saliency_maps = False
        explain_predicted_classes = True
        if explain_parameters is not None:
            update_progress_callback = explain_parameters.update_progress  # type: ignore
            process_saliency_maps = explain_parameters.process_saliency_maps
            explain_predicted_classes = explain_parameters.explain_predicted_classes

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

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

        self._init_task()

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

        recipe_cfg.work_dir = self._output_path
        recipe_cfg.resume = self._resume

        cfg = self.configure(recipe_cfg, self._model_ckpt, data_cfg, False, "test")

        samples_per_gpu = cfg.data.test_dataloader.get("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

        # Data loader
        mm_dataset = build_dataset(cfg.data.test)
        dataloader = build_dataloader(
            mm_dataset,
            samples_per_gpu=cfg.data.get("samples_per_gpu", 1),
            workers_per_gpu=cfg.data.get("workers_per_gpu", 0),
            num_gpus=len(cfg.gpu_ids),
            dist=cfg.distributed,
            seed=cfg.get("seed", None),
            shuffle=False,
        )

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.final
            if len(target_classes) < 1:
                raise KeyError(
                    f"target_classes={target_classes} is empty check the metadata from model ckpt or recipe "
                    "configuration"
                )
        else:
            target_classes = dataset.CLASSES

        # TODO: Check Inference FP16 Support
        model = build_detector(cfg)
        if getattr(cfg, "fp16", False):
            wrap_fp16_model(model)
        model.CLASSES = target_classes
        model.eval()
        feature_model = model.model_t if self._train_type == TrainType.SEMISUPERVISED else model
        model = build_data_parallel(model, cfg, distributed=False)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        time_monitor = None
        if cfg.get("custom_hooks", None):
            time_monitor = [hook.time_monitor for hook in cfg.custom_hooks if hook.type == "OTXProgressHook"]
            time_monitor = time_monitor[0] if time_monitor else None
        if time_monitor is not None:

            # pylint: disable=unused-argument
            def pre_hook(module, inp):
                time_monitor.on_test_batch_begin(None, None)

            def hook(module, inp, outp):
                time_monitor.on_test_batch_end(None, None)

            model.register_forward_pre_hook(pre_hook)
            model.register_forward_hook(hook)

        explainer = explain_parameters.explainer if explain_parameters else None
        explainer_hook = explainer_hook_selector.get(explainer.lower(), None)
        if explainer_hook is None:
            raise NotImplementedError(f"Explainer algorithm {explainer} not supported!")
        logger.info(f"Explainer algorithm: {explainer}")

        # Class-wise Saliency map for Single-Stage Detector, otherwise use class-ignore saliency map.
        eval_predictions = []
        with explainer_hook(feature_model) as saliency_hook:
            for data in dataloader:
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **data)
                eval_predictions.extend(result)
            saliency_maps = saliency_hook.records

        # Check and unwrap ImageTilingDataset object from TaskAdaptEvalDataset
        while hasattr(mm_dataset, "dataset") and not isinstance(mm_dataset, ImageTilingDataset):
            mm_dataset = mm_dataset.dataset

        # In the tiling case, select the first images which is map of the entire image
        if isinstance(mm_dataset, ImageTilingDataset):
            saliency_maps = [saliency_maps[i] for i in range(mm_dataset.num_samples)]

        outputs = dict(detections=eval_predictions, saliency_maps=saliency_maps)

        logger.info("run task done.")
        detections = outputs["detections"]
        explain_results = outputs["saliency_maps"]

        self._add_explanations_to_dataset(
            detections, explain_results, dataset, process_saliency_maps, explain_predicted_classes
        )
        logger.info("Explain completed")
        return dataset

    # This should be removed
    def update_override_configurations(self, config):
        """Update override_configs."""
        logger.info(f"update override config with: {config}")
        config = ConfigDict(**config)
        self.override_configs.update(config)

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

    # These need to be moved somewhere
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
