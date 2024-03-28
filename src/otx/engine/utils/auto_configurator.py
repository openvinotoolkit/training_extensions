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
from lightning.pytorch.cli import instantiate_class

from otx.core.config.data import DataModuleConfig, SamplerConfig, SubsetConfig, TileConfig
from otx.core.data.dataset.base import LabelInfo
from otx.core.data.module import OTXDataModule
from otx.core.model.entity.base import OVModel
from otx.core.types import PathLike
from otx.core.types.task import OTXTaskType
from otx.core.utils.imports import get_otx_root_path
from otx.core.utils.instantiators import partial_instantiate_class

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torchmetrics import Metric

    from otx.core.model.entity.base import OTXModel


logger = logging.getLogger()
RECIPE_PATH = get_otx_root_path() / "recipe"

DEFAULT_CONFIG_PER_TASK = {
    OTXTaskType.MULTI_CLASS_CLS: RECIPE_PATH / "classification" / "multi_class_cls" / "otx_efficientnet_b0.yaml",
    OTXTaskType.MULTI_LABEL_CLS: RECIPE_PATH / "classification" / "multi_label_cls" / "efficientnet_b0_light.yaml",
    OTXTaskType.H_LABEL_CLS: RECIPE_PATH / "classification" / "h_label_cls" / "efficientnet_b0_light.yaml",
    OTXTaskType.DETECTION: RECIPE_PATH / "detection" / "atss_mobilenetv2.yaml",
    OTXTaskType.ROTATED_DETECTION: RECIPE_PATH / "rotated_detection" / "maskrcnn_r50.yaml",
    OTXTaskType.SEMANTIC_SEGMENTATION: RECIPE_PATH / "semantic_segmentation" / "litehrnet_18.yaml",
    OTXTaskType.INSTANCE_SEGMENTATION: RECIPE_PATH / "instance_segmentation" / "maskrcnn_r50.yaml",
    OTXTaskType.ACTION_CLASSIFICATION: RECIPE_PATH / "action" / "action_classification" / "x3d.yaml",
    OTXTaskType.ACTION_DETECTION: RECIPE_PATH / "action" / "action_detection" / "x3d_fastrcnn.yaml",
    OTXTaskType.ANOMALY_CLASSIFICATION: RECIPE_PATH / "anomaly_classification" / "padim.yaml",
    OTXTaskType.ANOMALY_SEGMENTATION: RECIPE_PATH / "anomaly_segmentation" / "padim.yaml",
    OTXTaskType.ANOMALY_DETECTION: RECIPE_PATH / "anomaly_detection" / "padim.yaml",
    OTXTaskType.VISUAL_PROMPTING: RECIPE_PATH / "visual_prompting" / "sam_tiny_vit.yaml",
    OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING: RECIPE_PATH / "zero_shot_visual_prompting" / "sam_tiny_vit.yaml",
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
    "ava": [OTXTaskType.ACTION_DETECTION],
    "mvtec": [OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION],
}

OVMODEL_PER_TASK = {
    OTXTaskType.MULTI_CLASS_CLS: "otx.core.model.entity.classification.OVMulticlassClassificationModel",
    OTXTaskType.MULTI_LABEL_CLS: "otx.core.model.entity.classification.OVMultilabelClassificationModel",
    OTXTaskType.H_LABEL_CLS: "otx.core.model.entity.classification.OVHlabelClassificationModel",
    OTXTaskType.DETECTION: "otx.core.model.entity.detection.OVDetectionModel",
    OTXTaskType.ROTATED_DETECTION: "otx.core.model.entity.rotated_detection.OVRotatedDetectionModel",
    OTXTaskType.INSTANCE_SEGMENTATION: "otx.core.model.entity.instance_segmentation.OVInstanceSegmentationModel",
    OTXTaskType.SEMANTIC_SEGMENTATION: "otx.core.model.entity.segmentation.OVSegmentationModel",
    OTXTaskType.VISUAL_PROMPTING: "otx.core.model.entity.visual_prompting.OVVisualPromptingModel",
    OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING: "otx.core.model.entity.visual_prompting.OVZeroShotVisualPromptingModel",
    OTXTaskType.ACTION_CLASSIFICATION: "otx.core.model.entity.action_classification.OVActionClsModel",
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
    if len(data_format) > 1:
        logger.warning(f"Found multiple data formats: {data_format}. We will use the first one.")
    data_format = data_format[0]
    if data_format not in TASK_PER_DATA_FORMAT:
        msg = f"Can't find proper task. we are not support {data_format} format, yet."
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
        self.config["data"]["config"]["data_root"] = self.data_root
        data_config = deepcopy(self.config["data"]["config"])
        train_config = data_config.pop("train_subset")
        val_config = data_config.pop("val_subset")
        test_config = data_config.pop("test_subset")
        return OTXDataModule(
            task=self.config["data"]["task"],
            config=DataModuleConfig(
                train_subset=SubsetConfig(sampler=SamplerConfig(**train_config.pop("sampler", {})), **train_config),
                val_subset=SubsetConfig(sampler=SamplerConfig(**val_config.pop("sampler", {})), **val_config),
                test_subset=SubsetConfig(sampler=SamplerConfig(**test_config.pop("sampler", {})), **test_config),
                tile_config=TileConfig(**data_config.pop("tile_config", {})),
                **data_config,
            ),
        )

    def get_model(self, model_name: str | None = None, label_info: LabelInfo | None = None) -> OTXModel:
        """Retrieves the OTXModel instance based on the provided model name and meta information.

        Args:
            model_name (str | None): The name of the model to retrieve. If None, the default model will be used.
            label_info (LabelInfo | None): The meta information about the labels. If provided, the number of classes
                will be updated in the model's configuration.

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
        if model_name is not None:
            self._config = self._load_default_config(self.model_name)
        if label_info is not None:
            num_classes = label_info.num_classes
            self.config["model"]["init_args"]["num_classes"] = num_classes

            from otx.core.data.dataset.classification import HLabelInfo

            if isinstance(label_info, HLabelInfo):
                init_args = self.config["model"]["init_args"]
                init_args["num_multiclass_heads"] = label_info.num_multiclass_heads
                init_args["num_multilabel_classes"] = label_info.num_multilabel_classes

        logger.warning(f"Set Default Model: {self.config['model']}")
        return instantiate_class(args=(), init=self.config["model"])

    def get_optimizer(self) -> list[OptimizerCallable] | None:
        """Returns the optimizer callable based on the configuration.

        Returns:
            list[OptimizerCallable] | None: The optimizer callable.
        """
        optimizer_config = self.config.get("optimizer", None)
        logger.warning(f"Set Default Optimizer: {optimizer_config}")
        return partial_instantiate_class(init=optimizer_config)

    def get_scheduler(self) -> list[LRSchedulerCallable] | None:
        """Returns the instantiated scheduler based on the configuration.

        Returns:
            list[LRSchedulerCallable] | None: The instantiated scheduler.
        """
        scheduler_config = self.config.get("scheduler", None)
        logger.warning(f"Set Default Scheduler: {scheduler_config}")
        return partial_instantiate_class(init=scheduler_config)

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
            msg = f"{self.task} is not support OVModel."
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
        data_configuration = datamodule.config
        ov_test_config = self._load_default_config(model_name="openvino_model")["data"]["config"][f"{subset}_subset"]
        subset_config = getattr(data_configuration, f"{subset}_subset")
        subset_config.batch_size = ov_test_config["batch_size"]
        subset_config.transform_lib_type = ov_test_config["transform_lib_type"]
        subset_config.transforms = ov_test_config["transforms"]
        data_configuration.tile_config.enable_tiler = False
        msg = (
            f"For OpenVINO IR models, Update the following {subset} \n"
            f"\t transforms: {subset_config.transforms} \n"
            f"\t transform_lib_type: {subset_config.transform_lib_type} \n"
            f"\t batch_size: {subset_config.batch_size} \n"
            "And the tiler is disabled."
        )
        warn(msg, stacklevel=1)
        return OTXDataModule(task=datamodule.task, config=data_configuration)
