# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Auto-Configurator class & util functions for OTX Auto-Configuration."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import datumaro
from jsonargparse import ArgumentParser, Namespace

from otx.core.config.data import SamplerConfig, SubsetConfig, TileConfig, UnlabeledDataConfig, VisualPromptingConfig
from otx.core.data.module import OTXDataModule
from otx.core.model.base import OTXModel, OVModel
from otx.core.types import PathLike
from otx.core.types.label import LabelInfo, LabelInfoTypes
from otx.core.types.task import OTXTaskType
from otx.core.utils.imports import get_otx_root_path
from otx.core.utils.instantiators import partial_instantiate_class
from otx.utils.utils import can_pass_tile_config, get_model_cls_from_config, should_pass_label_info

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torchmetrics import Metric


logger = logging.getLogger()
RECIPE_PATH = get_otx_root_path() / "recipe"

DEFAULT_CONFIG_PER_TASK = {
    OTXTaskType.MULTI_CLASS_CLS: RECIPE_PATH / "classification" / "multi_class_cls" / "efficientnet_b0.yaml",
    OTXTaskType.MULTI_LABEL_CLS: RECIPE_PATH / "classification" / "multi_label_cls" / "efficientnet_b0.yaml",
    OTXTaskType.H_LABEL_CLS: RECIPE_PATH / "classification" / "h_label_cls" / "efficientnet_b0.yaml",
    OTXTaskType.DETECTION: RECIPE_PATH / "detection" / "atss_mobilenetv2.yaml",
    OTXTaskType.ROTATED_DETECTION: RECIPE_PATH / "rotated_detection" / "maskrcnn_r50.yaml",
    OTXTaskType.SEMANTIC_SEGMENTATION: RECIPE_PATH / "semantic_segmentation" / "litehrnet_18.yaml",
    OTXTaskType.INSTANCE_SEGMENTATION: RECIPE_PATH / "instance_segmentation" / "maskrcnn_r50.yaml",
    OTXTaskType.ACTION_CLASSIFICATION: RECIPE_PATH / "action_classification" / "x3d.yaml",
    OTXTaskType.ANOMALY_CLASSIFICATION: RECIPE_PATH / "anomaly_classification" / "padim.yaml",
    OTXTaskType.ANOMALY_SEGMENTATION: RECIPE_PATH / "anomaly_segmentation" / "padim.yaml",
    OTXTaskType.ANOMALY_DETECTION: RECIPE_PATH / "anomaly_detection" / "padim.yaml",
    OTXTaskType.VISUAL_PROMPTING: RECIPE_PATH / "visual_prompting" / "sam_tiny_vit.yaml",
    OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING: RECIPE_PATH / "zero_shot_visual_prompting" / "sam_tiny_vit.yaml",
    OTXTaskType.KEYPOINT_DETECTION: RECIPE_PATH / "keypoint_detection" / "rtmpose_tiny.yaml",
}

TASK_PER_DATA_FORMAT = {
    "imagenet_with_subset_dirs": [OTXTaskType.MULTI_CLASS_CLS, OTXTaskType.H_LABEL_CLS],
    "datumaro": [OTXTaskType.MULTI_LABEL_CLS],
    "coco_instances": [
        OTXTaskType.DETECTION,
        OTXTaskType.ROTATED_DETECTION,
        OTXTaskType.INSTANCE_SEGMENTATION,
        OTXTaskType.VISUAL_PROMPTING,
    ],
    "coco": [
        OTXTaskType.DETECTION,
        OTXTaskType.ROTATED_DETECTION,
        OTXTaskType.INSTANCE_SEGMENTATION,
        OTXTaskType.VISUAL_PROMPTING,
    ],
    "common_semantic_segmentation_with_subset_dirs": [OTXTaskType.SEMANTIC_SEGMENTATION],
    "kinetics": [OTXTaskType.ACTION_CLASSIFICATION],
    "mvtec": [OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION],
}

OVMODEL_PER_TASK = {
    OTXTaskType.MULTI_CLASS_CLS: "otx.core.model.classification.OVMulticlassClassificationModel",
    OTXTaskType.MULTI_LABEL_CLS: "otx.core.model.classification.OVMultilabelClassificationModel",
    OTXTaskType.H_LABEL_CLS: "otx.core.model.classification.OVHlabelClassificationModel",
    OTXTaskType.DETECTION: "otx.core.model.detection.OVDetectionModel",
    OTXTaskType.ROTATED_DETECTION: "otx.core.model.rotated_detection.OVRotatedDetectionModel",
    OTXTaskType.INSTANCE_SEGMENTATION: "otx.core.model.instance_segmentation.OVInstanceSegmentationModel",
    OTXTaskType.SEMANTIC_SEGMENTATION: "otx.core.model.segmentation.OVSegmentationModel",
    OTXTaskType.VISUAL_PROMPTING: "otx.core.model.visual_prompting.OVVisualPromptingModel",
    OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING: "otx.core.model.visual_prompting.OVZeroShotVisualPromptingModel",
    OTXTaskType.ACTION_CLASSIFICATION: "otx.core.model.action_classification.OVActionClsModel",
    OTXTaskType.ANOMALY_CLASSIFICATION: "otx.algo.anomaly.openvino_model.AnomalyOpenVINO",
    OTXTaskType.ANOMALY_DETECTION: "otx.algo.anomaly.openvino_model.AnomalyOpenVINO",
    OTXTaskType.ANOMALY_SEGMENTATION: "otx.algo.anomaly.openvino_model.AnomalyOpenVINO",
}


def configure_task(data_root: PathLike) -> OTXTaskType:
    """Configures the task based on the given data root.

    Args:
        data_root (PathLike): The root directory of the data.

    Returns:
        OTXTaskType: The configured task type, or None if data_root is None.

    Raises:
        ValueError: If the data format is not supported.
    """
    data_root = Path(data_root).resolve()

    data_format = datumaro.Environment().detect_dataset(str(data_root))
    if not len(data_format):
        msg = "Unable to detect data format."
        raise ValueError(msg)
    if len(data_format) > 1:
        logger.warning(f"Found multiple data formats: {data_format}. We will use the first one.")
    data_format = data_format[0]
    if data_format not in TASK_PER_DATA_FORMAT:
        msg = f"Can't find proper task. We do not support {data_format} format, yet."
        raise ValueError(msg)
    if len(TASK_PER_DATA_FORMAT[data_format]) > 1:
        logger.warning(
            f"Found multiple tasks with {data_format}: {TASK_PER_DATA_FORMAT[data_format]}. We will use the first one.",
        )
    return TASK_PER_DATA_FORMAT[data_format][0]


class AutoConfigurator:
    """This Class is used to configure the OTXDataModule, OTXModel, Optimizer, and Scheduler with OTX Default.

    Args:
        data_root (PathLike | None, optional): The root directory for data storage. Defaults to None.
        task (OTXTaskType | None, optional): The current task. Defaults to None.
        model_name (str | None, optional): Name of the model to use as the default.
            If None, the default model will be used. Defaults to None.

    Example:
        The following examples show how to use the AutoConfigurator class.

        >>> auto_configurator = AutoConfigurator(
        ...     data_root=<dataset/path>,
        ...     task=<OTXTaskType>,
        ... )

        # If task is None, the task will be configured based on the data root.
        >>> auto_configurator = AutoConfigurator(
        ...     data_root=<dataset/path>,
        ... )
    """

    def __init__(
        self,
        data_root: PathLike | None = None,
        task: OTXTaskType | None = None,
        model_name: str | None = None,
    ) -> None:
        self.data_root = data_root
        self._task = task
        self._config: dict | None = None
        self.model_name: str | None = model_name

    @property
    def task(self) -> OTXTaskType:
        """Returns the current task.

        Raises:
            RuntimeError: If there are no ready tasks.

        Returns:
            OTXTaskType: The current task.
        """
        if self._task is not None:
            return self._task
        if self.data_root is not None:
            self._task = configure_task(self.data_root)
            return self._task
        msg = "There are no ready task"
        raise RuntimeError(msg)

    @property
    def config(self) -> dict:
        """Retrieves the configuration for the auto configurator.

        If the configuration has not been loaded yet, it will be loaded using the default configuration
        based on the model name.

        Returns:
            dict: The configuration as a dict object.
        """
        if self._config is None:
            self._config = self._load_default_config(self.model_name)
        return self._config

    def _load_default_config(self, model_name: str | None = None) -> dict:
        """Load the default configuration for the specified model.

        Args:
            model_name (str | None): The name of the model. If provided, the configuration
                file name will be modified to use the specified model.

        Returns:
            dict: The loaded configuration.

        Raises:
            ValueError: If the task doesn't supported for auto-configuration.
        """
        config_file = DEFAULT_CONFIG_PER_TASK.get(self.task, None)
        if config_file is None:
            msg = f"{self.task} doesn't support Auto-Configuration."
            raise ValueError(msg)
        if model_name is not None:
            model_path = str(config_file).split("/")
            model_path[-1] = f"{model_name}.yaml"
            config_file = Path("/".join(model_path))
        from otx.cli.utils.jsonargparse import get_configuration

        return get_configuration(config_file)

    def get_datamodule(self) -> OTXDataModule | None:
        """Returns an instance of OTXDataModule with the configured data root.

        Returns:
            OTXDataModule | None: An instance of OTXDataModule.
        """
        if self.data_root is None:
            return None
        self.config["data"]["data_root"] = self.data_root
        data_config: dict = deepcopy(self.config["data"])
        train_config = data_config.pop("train_subset")
        val_config = data_config.pop("val_subset")
        test_config = data_config.pop("test_subset")
        unlabeled_config = data_config.pop("unlabeled_subset", {})
        tile_config = data_config.pop("tile_config", {})
        vpm_config = data_config.pop("vpm_config", {})

        _ = data_config.pop("__path__", {})  # Remove __path__ key that for CLI
        _ = data_config.pop("config", {})  # Remove config key that for CLI

        if data_config.get("adaptive_input_size") is not None:
            model_cls = get_model_cls_from_config(Namespace(self.config["model"]))
            data_config["input_size_multiplier"] = model_cls.input_size_multiplier

        return OTXDataModule(
            train_subset=SubsetConfig(sampler=SamplerConfig(**train_config.pop("sampler", {})), **train_config),
            val_subset=SubsetConfig(sampler=SamplerConfig(**val_config.pop("sampler", {})), **val_config),
            test_subset=SubsetConfig(sampler=SamplerConfig(**test_config.pop("sampler", {})), **test_config),
            unlabeled_subset=UnlabeledDataConfig(
                sampler=SamplerConfig(**unlabeled_config.pop("sampler", {})),
                **unlabeled_config,
            ),
            tile_config=TileConfig(**tile_config),
            vpm_config=VisualPromptingConfig(**vpm_config),
            **data_config,
        )

    def get_model(
        self,
        model_name: str | None = None,
        label_info: LabelInfoTypes | None = None,
        input_size: tuple[int, int] | int | None = None,
    ) -> OTXModel:
        """Retrieves the OTXModel instance based on the provided model name and meta information.

        Args:
            model_name (str | None): The name of the model to retrieve. If None, the default model will be used.
            label_info (LabelInfoTypes | None): The meta information about the labels.
                If provided, the number of classes will be updated in the model's configuration.
            input_size (tuple[int, int] | int | None, optional):
                Model input size in the order of height and width or a single integer for a side of a square.
                Defaults to None.

        Returns:
            OTXModel: The instantiated OTXModel instance.

        Example:
            The following examples show how to get the OTXModel class.

            # If model_name is None, the default model will be used from task.
            >>> auto_configurator.get_model(
            ...     label_info=<LabelInfo>,
            ... )

            # If model_name is str, the default config file is changed.
            >>> auto_configurator.get_model(
            ...     model_name=<model_name, str>,
            ...     label_info=<LabelInfo>,
            ... )
        """
        # TODO(vinnamki): There are some overlaps with src/otx/cli/cli.py::OTXCLI::instantiate_model
        if model_name is not None:
            self._config = self._load_default_config(self.model_name)

        skip = set()

        model_config = deepcopy(self.config["model"])

        if input_size is not None:
            model_config["init_args"]["input_size"] = (
                (input_size, input_size) if isinstance(input_size, int) else input_size
            )

        model_cls = get_model_cls_from_config(Namespace(model_config))

        if should_pass_label_info(model_cls):
            if label_info is None:
                msg = f"Given model class {model_cls} requires a valid label_info to instantiate."
                raise ValueError(msg)

            model_config["init_args"]["label_info"] = label_info
            skip.add("label_info")

        if can_pass_tile_config(model_cls) and (datamodule := self.get_datamodule()) is not None:
            model_config["init_args"]["tile_config"] = datamodule.tile_config
            skip.add("tile_config")

        model_parser = ArgumentParser()
        model_parser.add_subclass_arguments(
            OTXModel,
            "model",
            skip=skip,
            required=False,
            fail_untyped=False,
        )

        return model_parser.instantiate_classes(Namespace(model=model_config)).get("model")

    def get_optimizer(self) -> list[OptimizerCallable] | None:
        """Returns the optimizer callable based on the configuration.

        Returns:
            list[OptimizerCallable] | None: The optimizer callable.
        """
        if (
            (model_config := self.config.get("model", None))
            and (init_args := model_config.get("init_args", None))
            and (config := init_args.get("optimizer", None))
        ):
            if callable(config):
                return [config]
            return partial_instantiate_class(init=config)

        return None

    def get_scheduler(self) -> list[LRSchedulerCallable] | None:
        """Returns the instantiated scheduler based on the configuration.

        Returns:
            list[LRSchedulerCallable] | None: The instantiated scheduler.
        """
        if (
            (model_config := self.config.get("model", None))
            and (init_args := model_config.get("init_args", None))
            and (config := init_args.get("scheduler", None))
        ):
            if callable(config):
                return [config]
            return partial_instantiate_class(init=config)

        return None

    def get_metric(self) -> Metric | None:
        """Returns the instantiated metric based on the configuration.

        Returns:
            Metric | None: The instantiated metric.
        """
        if self.task in DEFAULT_CONFIG_PER_TASK:
            metric_config = self.config.get("metric", None)
            logger.warning(f"Set Default Metric: {metric_config}")

            # Currently, single metric only available.
            if metric_config:
                metric = partial_instantiate_class(init=metric_config)
                return metric[0] if isinstance(metric, list) else metric

        return None

    def get_ov_model(self, model_name: str, label_info: LabelInfo) -> OVModel:
        """Retrieves the OVModel instance based on the given model name and label information.

        Args:
            model_name (str): The name of the model.
            label_info (LabelInfo): The label information.

        Returns:
            OVModel: The OVModel instance.

        Raises:
            NotImplementedError: If the OVModel for the given task is not supported.
        """
        class_path = OVMODEL_PER_TASK.get(self.task, None)
        if class_path is None:
            msg = f"{self.task} doesn't support OVModel."
            raise NotImplementedError(msg)
        class_module, class_name = class_path.rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        ov_model = getattr(module, class_name)
        return ov_model(
            model_name=model_name,
            num_classes=label_info.num_classes,
        )

    def update_ov_subset_pipeline(self, datamodule: OTXDataModule, subset: str = "test") -> OTXDataModule:
        """Returns an OTXDataModule object with OpenVINO subset transforms applied.

        Args:
            datamodule (OTXDataModule): The original OTXDataModule object.
            subset (str, optional): The subset to update. Defaults to "test".

        Returns:
            OTXDataModule: The modified OTXDataModule object with OpenVINO subset transforms applied.
        """
        ov_config = self._load_default_config(model_name="openvino_model")["data"]
        subset_config = getattr(datamodule, f"{subset}_subset")
        subset_config.batch_size = ov_config[f"{subset}_subset"]["batch_size"]
        subset_config.transform_lib_type = ov_config[f"{subset}_subset"]["transform_lib_type"]
        subset_config.transforms = ov_config[f"{subset}_subset"]["transforms"]
        subset_config.to_tv_image = ov_config[f"{subset}_subset"]["to_tv_image"]
        datamodule.image_color_channel = ov_config["image_color_channel"]
        datamodule.tile_config.enable_tiler = False
        datamodule.unlabeled_subset.data_root = None
        msg = (
            f"For OpenVINO IR models, Update the following {subset} \n"
            f"\t transforms: {subset_config.transforms} \n"
            f"\t transform_lib_type: {subset_config.transform_lib_type} \n"
            f"\t batch_size: {subset_config.batch_size} \n"
            f"\t image_color_channel: {datamodule.image_color_channel} \n"
            "And the tiler is disabled."
        )
        warn(msg, stacklevel=1)
        return OTXDataModule(
            task=datamodule.task,
            data_format=datamodule.data_format,
            data_root=datamodule.data_root,
            train_subset=datamodule.train_subset,
            val_subset=datamodule.val_subset,
            test_subset=datamodule.test_subset,
            unlabeled_subset=datamodule.unlabeled_subset,
            tile_config=datamodule.tile_config,
            vpm_config=datamodule.vpm_config,
            image_color_channel=datamodule.image_color_channel,
            stack_images=datamodule.stack_images,
            include_polygons=datamodule.include_polygons,
            ignore_index=datamodule.ignore_index,
            unannotated_items_ratio=datamodule.unannotated_items_ratio,
            auto_num_workers=datamodule.auto_num_workers,
            device=datamodule.device,
        )
