"""Auto-Configurator class & util functions for OTX Auto-Configuration."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Union

import datumaro
from lightning.pytorch.cli import instantiate_class

from otx.core.config.data import DataModuleConfig, SubsetConfig, TilerConfig
from otx.core.data.dataset.base import LabelInfo
from otx.core.data.module import OTXDataModule
from otx.core.types.task import OTXTaskType
from otx.core.utils.imports import get_otx_root_path
from otx.core.utils.instantiators import partial_instantiate_class

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from typing_extensions import TypeAlias

    from otx.core.model.entity.base import OTXModel

PathLike: TypeAlias = Union[str, Path, os.PathLike]

logger = logging.getLogger()
RECIPE_PATH = get_otx_root_path() / "recipe"

DEFAULT_CONFIG_PER_TASK = {
    OTXTaskType.MULTI_CLASS_CLS: RECIPE_PATH / "classification" / "multi_class_cls" / "otx_efficientnet_b0.yaml",
    OTXTaskType.MULTI_LABEL_CLS: RECIPE_PATH / "classification" / "multi_label_cls" / "efficientnet_b0_light.yaml",
    OTXTaskType.DETECTION: RECIPE_PATH / "detection" / "atss_mobilenetv2.yaml",
    OTXTaskType.SEMANTIC_SEGMENTATION: RECIPE_PATH / "semantic_segmentation" / "litehrnet_18.yaml",
    OTXTaskType.INSTANCE_SEGMENTATION: RECIPE_PATH / "instance_segmentation" / "maskrcnn_r50.yaml",
    OTXTaskType.ACTION_CLASSIFICATION: RECIPE_PATH / "action" / "action_classification" / "x3d.yaml",
    OTXTaskType.ACTION_DETECTION: RECIPE_PATH / "action" / "action_detection" / "x3d_fastrcnn.yaml",
    OTXTaskType.ANOMALY_CLASSIFICATION: RECIPE_PATH / "anomaly" / "padim.yaml",
    OTXTaskType.VISUAL_PROMPTING: RECIPE_PATH / "visual_prompting" / "sam_tiny_vit.yaml",
    OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING: RECIPE_PATH / "zero_shot_visual_prompting" / "sam_tiny_vit.yaml",
}

TASK_PER_DATA_FORMAT = {
    "imagenet_with_subset_dirs": [OTXTaskType.MULTI_CLASS_CLS],
    "datumaro": [OTXTaskType.MULTI_LABEL_CLS],
    "coco_instances": [OTXTaskType.DETECTION, OTXTaskType.INSTANCE_SEGMENTATION, OTXTaskType.VISUAL_PROMPTING],
    "coco": [OTXTaskType.DETECTION, OTXTaskType.INSTANCE_SEGMENTATION, OTXTaskType.VISUAL_PROMPTING],
    "common_semantic_segmentation_with_subset_dirs": [OTXTaskType.SEMANTIC_SEGMENTATION],
    "kinetics": [OTXTaskType.ACTION_CLASSIFICATION],
    "ava": [OTXTaskType.ACTION_DETECTION],
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


def get_num_classes_from_meta_info(task: OTXTaskType, meta_info: LabelInfo) -> int:
    """Get the number of classes from the meta information.

    Args:
        task (OTXTaskType): The current task type.
        meta_info (LabelInfo): The meta information about the labels.

    Returns:
        int: The number of classes.
    """
    num_classes = len(meta_info.label_names)
    # Check background class
    if task in (OTXTaskType.SEMANTIC_SEGMENTATION):
        has_background = False
        for label in meta_info.label_names:
            if label.lower() == "background":
                has_background = True
                break
        if not has_background:
            num_classes += 1
    return num_classes


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
            ValueError: If the task is not supported for auto-configuration.
        """
        config_file = DEFAULT_CONFIG_PER_TASK.get(self.task, None)
        if config_file is None:
            msg = f"{self.task} is not support Auto-Configuration."
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
        data_config = self.config["data"]["config"].copy()
        return OTXDataModule(
            task=self.config["data"]["task"],
            config=DataModuleConfig(
                train_subset=SubsetConfig(**data_config.pop("train_subset")),
                val_subset=SubsetConfig(**data_config.pop("val_subset")),
                test_subset=SubsetConfig(**data_config.pop("test_subset")),
                tile_config=TilerConfig(**data_config.pop("tile_config", {})),
                **data_config,
            ),
        )

    def get_model(self, model_name: str | None = None, meta_info: LabelInfo | None = None) -> OTXModel:
        """Retrieves the OTXModel instance based on the provided model name and meta information.

        Args:
            model_name (str | None): The name of the model to retrieve. If None, the default model will be used.
            meta_info (LabelInfo | None): The meta information about the labels. If provided, the number of classes
                will be updated in the model's configuration.

        Returns:
            OTXModel: The instantiated OTXModel instance.

        Example:
            The following examples show how to get the OTXModel class.

            # If model_name is None, the default model will be used from task.
            >>> auto_configurator.get_model(
            ...     meta_info=<LabelInfo>,
            ... )

            # If model_name is str, the default config file is changed.
            >>> auto_configurator.get_model(
            ...     model_name=<model_name, str>,
            ...     meta_info=<LabelInfo>,
            ... )
        """
        if model_name is not None:
            self._config = self._load_default_config(self.model_name)
        if meta_info is not None:
            num_classes = get_num_classes_from_meta_info(self.task, meta_info)
            self.config["model"]["init_args"]["num_classes"] = num_classes
        logger.warning(f"Set Default Model: {self.config['model']}")
        return instantiate_class(args=(), init=self.config["model"])

    def get_optimizer(self) -> OptimizerCallable | None:
        """Returns the optimizer callable based on the configuration.

        Returns:
            OptimizerCallable | None: The optimizer callable.
        """
        optimizer_config = self.config.get("optimizer", None)
        logger.warning(f"Set Default Optimizer: {optimizer_config}")
        return partial_instantiate_class(init=optimizer_config)

    def get_scheduler(self) -> LRSchedulerCallable | None:
        """Returns the instantiated scheduler based on the configuration.

        Returns:
            LRSchedulerCallable | None: The instantiated scheduler.
        """
        scheduler_config = self.config.get("scheduler", None)
        logger.warning(f"Set Default Scheduler: {scheduler_config}")
        return partial_instantiate_class(init=scheduler_config)
