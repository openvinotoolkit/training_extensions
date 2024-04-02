"""Task of OTX Segmentation using mmsegmentation training backend."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import io
import math
import os
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import torch
from mmcv.runner import wrap_fp16_model
from mmcv.utils import Config, ConfigDict, get_git_hash
from mmseg import __version__
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import collect_env

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    BaseRecordingForwardHook,
    FeatureVectorHook,
)
from otx.algorithms.common.adapters.mmcv.utils import (
    adapt_batch_size,
    build_data_parallel,
    get_configs_by_pairs,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    InputSizeManager,
    OTXConfig,
)
from otx.algorithms.common.adapters.torch.utils import convert_sync_batchnorm
from otx.algorithms.common.configs.configuration_enums import BatchSizeAdaptType
from otx.algorithms.common.configs.training_base import TrainType
from otx.algorithms.common.tasks.nncf_task import NNCFBaseTask
from otx.algorithms.common.utils import is_hpu_available
from otx.algorithms.common.utils.data import get_dataset
from otx.algorithms.segmentation.adapters.mmseg.apis.train import train_segmentor
from otx.algorithms.segmentation.adapters.mmseg.configurer import (
    IncrSegmentationConfigurer,
    SegmentationConfigurer,
    SemiSLSegmentationConfigurer,
)
from otx.algorithms.segmentation.adapters.mmseg.utils.builder import build_segmentor
from otx.algorithms.segmentation.adapters.mmseg.utils.exporter import (
    SegmentationExporter,
)
from otx.algorithms.segmentation.task import OTXSegmentationTask
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import ids_to_strings
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import (
    ModelEntity,
    ModelPrecision,
)
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.utils.logger import get_logger

if is_hpu_available():
    import habana_frameworks.torch.core as htcore

logger = get_logger()

# TODO Remove unnecessary pylint disable
# pylint: disable=too-many-lines


class MMSegmentationTask(OTXSegmentationTask):
    """Task class for OTX segmentation using mmsegmentation training backend."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        super().__init__(task_environment, output_path)
        self._data_cfg: Optional[Config] = None
        self._recipe_cfg: Optional[Config] = None

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _init_task(self):  # noqa
        """Initialize task."""
        self._recipe_cfg = OTXConfig.fromfile(os.path.join(self._model_dir, "model.py"))
        self._recipe_cfg.domain = self._task_type.domain
        self._config = self._recipe_cfg

        self.set_seed()

        logger.info("initialized.")

    # pylint: disable=too-many-arguments
    def configure(
        self,
        training=True,
        ir_options=None,
        export=False,
    ):
        """Patch mmcv configs for OTX segmentation settings."""

        # deepcopy all configs to make sure
        # changes under Configuerer and below does not take an effect to OTX for clear distinction
        recipe_cfg = deepcopy(self._recipe_cfg)
        assert recipe_cfg is not None, "'recipe_cfg' is not initialized."

        if self._data_cfg is not None:
            data_classes = [label.name for label in self._labels]
        else:
            data_classes = None
        model_classes = [label.name for label in self._model_label_schema]

        recipe_cfg.work_dir = self._output_path
        recipe_cfg.resume = self._resume

        if self._train_type == TrainType.Incremental:
            configurer = IncrSegmentationConfigurer(
                "segmentation",
                training,
                export,
                self.override_configs,
                self.on_hook_initialized,
                self._time_monitor,
                self._learning_curves,
            )
        elif self._train_type == TrainType.Semisupervised:
            configurer = SemiSLSegmentationConfigurer(
                "segmentation",
                training,
                export,
                self.override_configs,
                self.on_hook_initialized,
                self._time_monitor,
                self._learning_curves,
            )
        else:
            configurer = SegmentationConfigurer(
                "segmentation",
                training,
                export,
                self.override_configs,
                self.on_hook_initialized,
                self._time_monitor,
                self._learning_curves,
            )
        cfg = configurer.configure(
            recipe_cfg,
            self.data_pipeline_path,
            self._hyperparams,
            self._model_ckpt,
            self._data_cfg,
            ir_options,
            data_classes,
            model_classes,
            self._input_size,
        )
        self._config = cfg
        self._input_size = cfg.model.pop("input_size", None)

        return cfg

    def build_model(
        self,
        cfg: Config,
        fp16: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        """Build model from model_builder."""
        model_builder = getattr(self, "model_builder", build_segmentor)
        model = model_builder(cfg, **kwargs)
        if bool(fp16):
            wrap_fp16_model(model)
        return model

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

        self._init_task()

        cfg = self.configure(False, None)
        logger.info("infer!")

        # FIXME: Currently segmentor does not support multi batch inference.
        if "test" in cfg.data and "test_dataloader" in cfg.data:
            cfg.data.test_dataloader["samples_per_gpu"] = 1

        # Data loader
        mm_dataset = build_dataset(cfg.data.test)
        dataloader = build_dataloader(
            mm_dataset,
            samples_per_gpu=cfg.data.test_dataloader.get("samples_per_gpu", 1),
            workers_per_gpu=cfg.data.test_dataloader.get("workers_per_gpu", 0),
            num_gpus=len(cfg.gpu_ids),
            dist=cfg.distributed,
            seed=cfg.get("seed", None),
            persistent_workers=False,
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
        feature_model = model.model_s if self._train_type == TrainType.Semisupervised else model
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

        if not dump_features:
            feature_vector_hook: Union[nullcontext, BaseRecordingForwardHook] = nullcontext()
        else:
            feature_vector_hook = FeatureVectorHook(feature_model)

        with feature_vector_hook:
            for data in dataloader:
                with torch.no_grad():
                    result = model(return_loss=False, output_logits=True, **data)
                eval_predictions.append(result)
                if isinstance(feature_vector_hook, nullcontext):
                    feature_vectors = [None] * len(mm_dataset)
                else:
                    feature_vectors = feature_vector_hook.records

        assert len(eval_predictions) == len(feature_vectors), (
            "Number of elements should be the same, however, number of outputs are ",
            f"{len(eval_predictions)} and {len(feature_vectors)}",
        )

        outputs = dict(
            classes=target_classes,
            eval_predictions=eval_predictions,
            feature_vectors=feature_vectors,
        )
        return outputs

    # pylint: disable=too-many-branches, too-many-statements
    def _train_model(
        self,
        dataset: DatasetEntity,
    ):
        """Train function in MMSegmentationTask."""
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

        cfg = self.configure(True, None)
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

        if self._train_type == TrainType.Semisupervised:
            # forward the knowledge of num iters per epoch to model for filter loss
            bs_per_gpu = cfg.data.train_dataloader["samples_per_gpu"]
            actual_bs = bs_per_gpu * torch.distributed.get_world_size() if cfg.distributed else bs_per_gpu
            cfg.model.num_iters_per_epoch = math.ceil(len(datasets[0]) / actual_bs)

        # FIXME: Currently segmentor does not support multi batch evaluation.
        # For the Self-SL case, there is no val data. So, need to check the

        if "val" in cfg.data and "val_dataloader" in cfg.data:
            cfg.data.val_dataloader["samples_per_gpu"] = 1

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.final
        else:
            target_classes = datasets[0].CLASSES

        # Metadata
        meta = dict()
        meta["env_info"] = env_info
        meta["seed"] = cfg.seed
        meta["exp_name"] = cfg.work_dir
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmseg_version=__version__ + get_git_hash()[:7],
                CLASSES=target_classes,
            )

        # Model
        model = self.build_model(cfg, fp16=cfg.get("fp16", False), is_training=self._is_training)
        model.train()
        model.CLASSES = target_classes

        if is_hpu_available():
            # TODO (sungchul): move it to appropriate location if needed
            htcore.hpu.ModuleCacher(max_graphs=10)(model=model.backbone, inplace=True)
            htcore.hpu.ModuleCacher(max_graphs=10)(model=model.decode_head, inplace=True)

        validate = bool(cfg.data.get("val", None))

        if self._hyperparams.learning_parameters.auto_adapt_batch_size != BatchSizeAdaptType.NONE:
            is_nncf = isinstance(self, NNCFBaseTask)
            adapt_batch_size(
                train_segmentor,
                model,
                datasets,
                cfg,
                cfg.distributed,
                is_nncf,
                meta=meta,
                not_increase=(self._hyperparams.learning_parameters.auto_adapt_batch_size == BatchSizeAdaptType.SAFE),
                model_builder=getattr(self, "model_builder") if is_nncf else None,
            )

        if cfg.distributed:
            convert_sync_batchnorm(model)

        train_segmentor(
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
        best_ckpt_path = glob.glob(os.path.join(cfg.work_dir, "best_mDice_*.pth"))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        best_ckpt_path = glob.glob(os.path.join(cfg.work_dir, "best_mIoU_*.pth"))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        return dict(
            final_ckpt=output_ckpt_path,
        )

    def _explain_model(self):
        """Explain function of OTX Segmentation Task."""
        raise NotImplementedError

    # pylint: disable=too-many-statements
    def _export_model(
        self,
        precision: ModelPrecision = ModelPrecision.FP32,
        export_format: ExportType = ExportType.ONNX,
        dump_features: bool = True,
    ):
        """Export function of OTX Segmentation Task."""
        # copied from OTX inference_task.py
        self._data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    otx_dataset=None,
                    labels=self._labels,
                ),
                test=ConfigDict(
                    otx_dataset=None,
                    labels=self._labels,
                ),
            )
        )
        self._init_task()

        cfg = self.configure(False, None, export=True)

        self._precision[0] = precision
        export_options: Dict[str, Any] = {}
        export_options["deploy_cfg"] = self._init_deploy_cfg(cfg)
        assert len(self._precision) == 1
        export_options["precision"] = str(self._precision[0])
        export_options["type"] = str(export_format)

        export_options["deploy_cfg"]["dump_features"] = dump_features
        if dump_features:
            output_names = export_options["deploy_cfg"]["ir_config"]["output_names"]
            if "feature_vector" not in output_names:
                output_names.append("feature_vector")
            if export_options["deploy_cfg"]["codebase_config"]["task"] != "Segmentation":
                if "saliency_map" not in output_names:
                    output_names.append("saliency_map")
        export_options["model_builder"] = getattr(self, "model_builder", build_segmentor)

        if self._precision[0] == ModelPrecision.FP16:
            export_options["deploy_cfg"]["backend_config"]["mo_options"]["flags"].append("--compress_to_fp16")

        backend_cfg_backup = {}
        if export_format == ExportType.ONNX:
            backend_cfg_backup = export_options["deploy_cfg"]["backend_config"]
            export_options["deploy_cfg"]["backend_config"] = {"type": "onnxruntime"}
            export_options["deploy_cfg"]["ir_config"]["dynamic_axes"]["input"] = {0: "batch"}

        exporter = SegmentationExporter()
        results = exporter.run(
            cfg,
            **export_options,
        )

        if export_format == ExportType.ONNX:
            results["inference_parameters"] = {}
            results["inference_parameters"]["mean_values"] = " ".join(
                map(str, backend_cfg_backup["mo_options"]["args"]["--mean_values"])
            )
            results["inference_parameters"]["scale_values"] = " ".join(
                map(str, backend_cfg_backup["mo_options"]["args"]["--scale_values"])
            )

        return results

    # This should moved somewhere
    def _init_deploy_cfg(self, cfg: Config) -> Union[Config, None]:
        base_dir = os.path.abspath(os.path.dirname(self._task_environment.model_template.model_template_path))
        deploy_cfg_path = os.path.join(base_dir, "deployment.py")
        deploy_cfg = None
        if os.path.exists(deploy_cfg_path):
            deploy_cfg = OTXConfig.fromfile(deploy_cfg_path)

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

            def patch_input_shape(deploy_cfg):
                input_size_manager = InputSizeManager(cfg)
                size = input_size_manager.get_input_size_from_cfg("test")
                assert all(isinstance(i, int) and i > 0 for i in size)
                # default is static shape to prevent an unexpected error
                # when converting to OpenVINO IR
                deploy_cfg.backend_config.model_inputs = [ConfigDict(opt_shapes=ConfigDict(input=[1, 3, *size]))]

            patch_input_preprocessing(deploy_cfg)
            patch_input_shape(deploy_cfg)

        return deploy_cfg

    # This should be removed
    def update_override_configurations(self, config):
        """Update override_configs."""
        logger.info(f"update override config with: {config}")
        config = ConfigDict(**config)
        self.override_configs.update(config)

    def save_model(self, output_model: ModelEntity):
        """Save best model weights in SegmentationTrainTask."""
        logger.info("called save_model")
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {
            "model": model_ckpt,
            "config": hyperparams_str,
            "labels": labels,
            "input_size": self._input_size,
            "VERSION": 1,
        }

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        output_model.precision = self._precision
