"""Task of OTX Video Recognition using mmaction training backend."""

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
import os
import time
from copy import deepcopy
from functools import partial
from typing import Dict, Optional, Union

import torch
from mmaction import __version__
from mmaction.apis import train_model
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model as build_videomodel
from mmaction.utils import collect_env
from mmcv.runner import CheckpointLoader, load_checkpoint, wrap_fp16_model
from mmcv.utils import Config, ConfigDict, ProgressBar, get_git_hash
from torch import distributed as dist

from otx.algorithms.action.adapters.mmaction import (
    Exporter,
)
from otx.algorithms.action.task import OTXActionTask
from otx.algorithms.common.adapters.mmcv.utils import (
    adapt_batch_size,
    build_data_parallel,
    get_configs_by_pairs,
    patch_adaptive_interval_training,
    patch_data_pipeline,
    patch_early_stopping,
    patch_from_hyperparams,
    patch_persistent_workers,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    MPAConfig,
    update_or_add_custom_hook,
)
from otx.algorithms.common.configs.configuration_enums import BatchSizeAdaptType
from otx.algorithms.common.utils import append_dist_rank_suffix
from otx.algorithms.common.utils.data import get_dataset
from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import ModelPrecision
from otx.api.entities.model_template import TaskType
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.core.data import caching

logger = get_logger()

# TODO Remove unnecessary pylint disable
# pylint: disable=too-many-lines


class MMActionTask(OTXActionTask):
    """Task class for OTX action using mmaction training backend."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        super().__init__(task_environment, output_path)
        self._data_cfg: Optional[Config] = None
        self._recipe_cfg: Optional[Config] = None

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _init_task(self, export: bool = False):  # noqa
        """Initialize task."""

        self._recipe_cfg = MPAConfig.fromfile(os.path.join(self._model_dir, "model.py"))
        self._recipe_cfg.domain = self._task_type.domain
        self._config = self._recipe_cfg

        self.set_seed()

        # Belows may go to the configure function
        patch_data_pipeline(self._recipe_cfg, self.data_pipeline_path)

        if not export:
            patch_from_hyperparams(self._recipe_cfg, self._hyperparams)
            self._recipe_cfg.total_epochs = self._recipe_cfg.runner.max_epochs

        if "custom_hooks" in self.override_configs:
            override_custom_hooks = self.override_configs.pop("custom_hooks")
            for override_custom_hook in override_custom_hooks:
                update_or_add_custom_hook(self._recipe_cfg, ConfigDict(override_custom_hook))
        if len(self.override_configs) > 0:
            logger.info(f"before override configs merging = {self._recipe_cfg}")
            self._recipe_cfg.merge_from_dict(self.override_configs)
            logger.info(f"after override configs merging = {self._recipe_cfg}")

        # add Cancel training hook
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

        # Update recipe with caching modules
        self._update_caching_modules(self._recipe_cfg.data)

        logger.info("initialized.")

    def build_model(
        self,
        cfg: Config,
        fp16: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        """Build model from model_builder."""
        model_builder = getattr(self, "model_builder", build_videomodel)
        model = model_builder(cfg.model, **kwargs)

        checkpoint = cfg.pop("load_from", None)
        if checkpoint is not None:
            load_checkpoint(model, checkpoint, map_location="cpu")
        cfg.load_from = checkpoint

        if fp16:
            wrap_fp16_model(model)

        return model

    # pylint: disable=too-many-arguments
    def configure(
        self,
        training=True,
        subset="train",
        ir_options=None,
    ):
        """Patch mmcv configs for OTX action settings."""

        # deepcopy all configs to make sure
        # changes under MPA and below does not take an effect to OTX for clear distinction
        recipe_cfg = deepcopy(self._recipe_cfg)
        assert recipe_cfg is not None, "'recipe_cfg' is not initialized."

        recipe_cfg.work_dir = self._output_path
        recipe_cfg.resume = self._resume
        recipe_cfg.omnisource = False

        self._configure_device(recipe_cfg, training)

        if self._data_cfg is not None:
            recipe_cfg.merge_from_dict(self._data_cfg)

        if self._task_type == TaskType.ACTION_CLASSIFICATION:
            _dataset_type = "OTXActionClsDataset"
        else:
            _dataset_type = "OTXActionDetDataset"
        for subset in ("train", "val", "test", "unlabeled"):
            _cfg = recipe_cfg.data.get(subset, None)
            if not _cfg:
                continue
            _cfg.type = _dataset_type
            while "dataset" in _cfg:
                _cfg = _cfg.dataset
            _cfg.labels = self._labels

        if self._task_type == TaskType.ACTION_CLASSIFICATION:
            recipe_cfg.model["cls_head"].num_classes = len(self._labels)
        elif self._task_type == TaskType.ACTION_DETECTION:
            recipe_cfg.model["roi_head"]["bbox_head"].num_classes = len(self._labels) + 1
            if len(self._labels) < 5:
                recipe_cfg.model["roi_head"]["bbox_head"]["topk"] = len(self._labels) - 1

        recipe_cfg.data.videos_per_gpu = recipe_cfg.data.pop("samples_per_gpu", None)

        patch_adaptive_interval_training(recipe_cfg)
        patch_early_stopping(recipe_cfg)
        patch_persistent_workers(recipe_cfg)

        if self._model_ckpt is not None:
            recipe_cfg.load_from = self.get_model_ckpt(self._model_ckpt)
            if self._resume:  # after updating to mmaction 1.x, need to be removed
                recipe_cfg.resume_from = recipe_cfg.load_from

        self._config = recipe_cfg
        return recipe_cfg

    @staticmethod
    def get_model_ckpt(ckpt_path, new_path=None):
        """Get pytorch model weights."""
        ckpt = CheckpointLoader.load_checkpoint(ckpt_path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
            if not new_path:
                new_path = ckpt_path[:-3] + "converted.pth"
            new_path = append_dist_rank_suffix(new_path)
            torch.save(ckpt, new_path)
            return new_path
        return ckpt_path

    def _configure_device(self, cfg: Config, training: bool):
        """Setting device for training and inference."""
        cfg.distributed = False
        if torch.distributed.is_initialized():
            cfg.gpu_ids = [int(os.environ["LOCAL_RANK"])]
            if training:  # TODO multi GPU is available only in training. Evaluation needs to be supported later.
                cfg.distributed = True
                self.configure_distributed(cfg)
        elif "gpu_ids" not in cfg:
            cfg.gpu_ids = range(1)

        # consider "cuda" and "cpu" device only
        if not torch.cuda.is_available():
            cfg.device = "cpu"
            cfg.gpu_ids = range(-1, 0)
        else:
            cfg.device = "cuda"

    @staticmethod
    def configure_distributed(cfg: Config):
        """Patching for distributed training."""
        if hasattr(cfg, "dist_params") and cfg.dist_params.get("linear_scale_lr", False):
            new_lr = dist.get_world_size() * cfg.optimizer.lr
            logger.info(
                f"enabled linear scaling rule to the learning rate. \
                changed LR from {cfg.optimizer.lr} to {new_lr}"
            )
            cfg.optimizer.lr = new_lr

    # pylint: disable=too-many-branches, too-many-statements
    def _train_model(
        self,
        dataset: DatasetEntity,
    ):
        """Train function in MMActionTask."""
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

        self._is_training = True

        self._init_task()

        cfg = self.configure(True, "train", None)
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

        # FIXME: Currently action do not support multi batch evaluation. This will be fixed
        if "val" in cfg.data:
            cfg.data.val_dataloader["videos_per_gpu"] = 1

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.get("final", [])
        else:
            target_classes = datasets[0].CLASSES

        # Metadata
        meta = dict()
        meta["env_info"] = env_info
        meta["seed"] = cfg.seed
        meta["exp_name"] = cfg.work_dir
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmaction2_version=__version__ + get_git_hash()[:7],
                CLASSES=target_classes,
            )

        # Model
        model = self.build_model(cfg, fp16=cfg.get("fp16", False))
        model.train()
        model.CLASSES = target_classes

        if cfg.distributed:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        validate = bool(cfg.data.get("val", None))

        if self._hyperparams.learning_parameters.auto_adapt_batch_size != BatchSizeAdaptType.NONE:
            train_func = partial(train_model, meta=deepcopy(meta), model=deepcopy(model), distributed=False)
            adapt_batch_size(
                train_func,
                cfg,
                datasets,
                validate,
                not_increase=(self._hyperparams.learning_parameters.auto_adapt_batch_size == BatchSizeAdaptType.SAFE),
            )

        train_model(
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
        if best_ckpt_path:
            output_ckpt_path = best_ckpt_path[0]
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

        dump_features = False
        dump_saliency_map = False

        self._init_task()

        cfg = self.configure(False, "test", None)
        logger.info("infer!")

        videos_per_gpu = cfg.data.test_dataloader.get("videos_per_gpu", 1)

        # Data loader
        mm_dataset = build_dataset(cfg.data.test)
        dataloader = build_dataloader(
            mm_dataset,
            videos_per_gpu=videos_per_gpu,
            workers_per_gpu=cfg.data.test_dataloader.get("workers_per_gpu", 0),
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
            target_classes = mm_dataset.CLASSES

        # Model
        model = self.build_model(cfg, fp16=cfg.get("fp16", False))
        model.CLASSES = target_classes
        model.eval()
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

        eval_predictions = []
        feature_vectors = []
        saliency_maps = []

        def dump_features_hook():
            raise NotImplementedError("get_feature_vector function for mmaction is not implemented")

        # pylint: disable=unused-argument
        def dummy_dump_features_hook(model, inp, out):
            feature_vectors.append(None)

        def dump_saliency_hook():
            raise NotImplementedError("get_saliency_map for mmaction is not implemented")

        # pylint: disable=unused-argument
        def dummy_dump_saliency_hook(model, inp, out):
            saliency_maps.append(None)

        feature_vector_hook = dump_features_hook if dump_features else dummy_dump_features_hook
        saliency_map_hook = dump_saliency_hook if dump_saliency_map else dummy_dump_saliency_hook

        prog_bar = ProgressBar(len(dataloader))
        with model.module.backbone.register_forward_hook(feature_vector_hook):
            with model.module.backbone.register_forward_hook(saliency_map_hook):
                for data in dataloader:
                    with torch.no_grad():
                        result = model(return_loss=False, **data)
                    eval_predictions.extend(result)
                    for _ in range(videos_per_gpu):
                        prog_bar.update()
        prog_bar.file.write("\n")

        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"]:
            cfg.evaluation.pop(key, None)

        metric = None
        metric_name = self._recipe_cfg.evaluation.final_metric
        if inference_parameters:
            if inference_parameters.is_evaluation:
                metric = mm_dataset.evaluate(eval_predictions, **self._recipe_cfg.evaluation)[metric_name]

        assert len(eval_predictions) == len(feature_vectors), f"{len(eval_predictions)} != {len(feature_vectors)}"
        assert len(eval_predictions) == len(saliency_maps), f"{len(eval_predictions)} != {len(saliency_maps)}"
        predictions = zip(eval_predictions, feature_vectors, saliency_maps)

        return predictions, metric

    def _export_model(self, precision: ModelPrecision, export_format: ExportType, dump_features: bool):
        """Main export function."""
        self._data_cfg = None
        self._init_task(export=True)

        cfg = self.configure(False, "test", None)
        deploy_cfg = self._init_deploy_cfg(cfg)

        state_dict = torch.load(self._model_ckpt)
        if "model" in state_dict.keys():
            state_dict = state_dict["model"]
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]

        self._precision[0] = precision
        half_precision = precision == ModelPrecision.FP16

        exporter = Exporter(
            cfg,
            state_dict,
            deploy_cfg,
            f"{self._output_path}/openvino",
            half_precision,
            onnx_only=export_format == ExportType.ONNX,
        )
        exporter.export()

        results: Dict[str, Dict[str, str]] = {"outputs": {}}

        if export_format == ExportType.ONNX:
            onnx_file = [f for f in os.listdir(self._output_path) if f.endswith(".onnx")][0]
            results["outputs"]["onnx"] = os.path.join(self._output_path, onnx_file)
        else:
            bin_file = [f for f in os.listdir(self._output_path) if f.endswith(".bin")][0]
            xml_file = [f for f in os.listdir(self._output_path) if f.endswith(".xml")][0]
            results["outputs"]["bin"] = os.path.join(self._output_path, bin_file)
            results["outputs"]["xml"] = os.path.join(self._output_path, xml_file)

        return results

    # This should be removed
    def update_override_configurations(self, config):
        """Update override_configs."""
        logger.info(f"update override config with: {config}")
        config = ConfigDict(**config)
        self.override_configs.update(config)

    # This should moved somewhere
    def _init_deploy_cfg(self, cfg: Config) -> Union[Config, None]:
        base_dir = os.path.abspath(os.path.dirname(self._task_environment.model_template.model_template_path))
        deploy_cfg_path = os.path.join(base_dir, "deployment.py")
        deploy_cfg = None
        if os.path.exists(deploy_cfg_path):
            deploy_cfg = MPAConfig.fromfile(deploy_cfg_path)

            def patch_input_preprocessing(deploy_cfg):
                normalize_cfg = get_configs_by_pairs(
                    cfg.data.test.pipeline,
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

            patch_input_preprocessing(deploy_cfg)
            if not deploy_cfg.backend_config.get("model_inputs", []):
                raise NotImplementedError("Video recognition task must specify model input info in deployment.py")

        return deploy_cfg

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
