"""Auto-Configurator class & util functions for OTX Auto-Configuration."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import datumaro
from lightning.pytorch.cli import instantiate_class

from otx.core.config.data import DataModuleConfig, SubsetConfig
from otx.core.data.module import OTXDataModule
from otx.core.model.entity.base import OTXModel
from otx.core.types.task import OTXTaskType
from otx.core.utils import get_otx_root_path
from otx.core.utils.instantiators import partial_instantiate_class

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable


logger = logging.getLogger()
RECIPE_PATH = get_otx_root_path() / "recipe"

DEFAULT_CONFIG_PER_TASK = {
    OTXTaskType.MULTI_CLASS_CLS: RECIPE_PATH / "classification" / "multi_class_cls" / "otx_efficientnet_b0.yaml",
    OTXTaskType.MULTI_LABEL_CLS: RECIPE_PATH / "classification" / "multi_label_cls" / "efficientnet_b0_light.yaml",
    OTXTaskType.H_LABEL_CLS: RECIPE_PATH / "classification" / "h_label_cls" / "efficientnet_b0_light.yaml",
    OTXTaskType.DETECTION: RECIPE_PATH / "detection" / "atss_mobilenetv2.yaml",
    OTXTaskType.SEMANTIC_SEGMENTATION: RECIPE_PATH / "semantic_segmentation" / "litehrnet_18.yaml",
    OTXTaskType.INSTANCE_SEGMENTATION: RECIPE_PATH / "instance_segmentation" / "maskrcnn_r50.yaml",
    OTXTaskType.ACTION_CLASSIFICATION: RECIPE_PATH / "action" / "action_classification" / "x3d.yaml",
    OTXTaskType.ACTION_DETECTION: RECIPE_PATH / "action" / "action_detection" / "x3d_fastrcnn.yaml",
    OTXTaskType.VISUAL_PROMPTING: RECIPE_PATH / "visual_prompting" / "sam_tiny_vit.yaml",
}


def is_cvat_format(path: Path) -> bool:
    """Detect whether data path is CVAT format or not.

    Currently, we used multi-video CVAT format for Action tasks.

    This function can detect the multi-video CVAT format.

    Multi-video CVAT format
    root
    |--video_0
        |--images
            |--frame0001.png
        |--annotations.xml
    |--video_1
    |--video_2

    will be deprecated soon.
    """
    cvat_format = sorted(["images", "annotations.xml"])
    for sub_folder in path.iterdir():
        # video_0, video_1, ...
        sub_folder_path = path / sub_folder
        # files must be same with cvat_format
        if sub_folder_path.is_dir():
            files = sorted(sub_folder_path.iterdir())
            if files != cvat_format:
                return False
    return True


def is_mvtec_format(path: Path) -> bool:
    """Detect whether data path is MVTec format or not.

    Check the first-level architecture folder, to know whether the dataset is MVTec or not.

    MVTec default structure like as below:
    root
    |--ground_truth
    |--train
    |--test

    will be deprecated soon.
    """
    mvtec_format = sorted(["ground_truth", "train", "test"])
    folder_list = []
    for sub_folder in path.iterdir():
        sub_folder_path = path / sub_folder
        # only use the folder name.
        if sub_folder_path.is_dir():
            folder_list.append(sub_folder)
    return sorted(folder_list) == mvtec_format


def configure_task(data_root: str | Path) -> OTXTaskType:
    """Find the format of dataset."""
    mapping_task_type_to_data_format = {
        OTXTaskType.MULTI_CLASS_CLS: ["imagenet_with_subset_dirs"],
        OTXTaskType.MULTI_LABEL_CLS: ["datumaro"],
        OTXTaskType.DETECTION: ["coco_instances", "voc", "yolo"],
        OTXTaskType.SEMANTIC_SEGMENTATION: [
            "cityscapes",
            "common_semantic_segmentation_with_subset_dirs",
            "voc",
            "ade20k2017",
            "ade20k2020",
        ],
        OTXTaskType.INSTANCE_SEGMENTATION: ["coco_instances", "voc"],
        OTXTaskType.ACTION_CLASSIFICATION: ["kinetics"],
        OTXTaskType.ACTION_DETECTION: ["ava"],
        OTXTaskType.VISUAL_PROMPTING: ["coco_instances"],
    }
    data_root = Path(data_root).resolve()

    data_format: str = ""

    # Currently, below `if/else` statements is mandatory
    # because Datumaro can't detect the multi-cvat and mvtec.
    # After, the upgrade of Datumaro, below codes will be changed.
    if is_cvat_format(data_root):
        data_format = "multi-cvat"
    elif is_mvtec_format(data_root):
        data_format = "mvtec"
    else:
        data_formats = datumaro.Environment().detect_dataset(str(data_root))
        if "imagenet_with_subset_dirs" not in data_formats:
            data_format = data_formats[0]
        else:
            data_format = "imagenet_with_subset_dirs"
    for task_key, data_value in mapping_task_type_to_data_format.items():
        if data_format in data_value:
            return task_key
    msg = f"Can't find proper task. we are not support {data_format} format, yet."
    raise ValueError(msg)


class AutoConfigurator:
    """Class responsible for auto configuration of data and model in OTX.

    Args:
        data_root (str | Path | None, optional): The root directory for data storage. Defaults to None.
        task (OTXTaskType | None, optional): The current task. Defaults to None.

    Attributes:
        data_root (str | Path): The root directory for data storage.
        task (OTXTaskType): The current task.
        model_cfg (DictConfig): The model configuration.

    Raises:
        RuntimeError: If the data root is not set.
        ValueError: If the task is not supported.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        task: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.data_root = data_root
        self._task = self._configure_task_type(data_root, task)
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
        if self._task is None:
            msg = "There are no ready task"
            raise RuntimeError(msg)
        return self._task

    @property
    def config(self) -> dict:
        """Retrieves the configuration for the auto configurator.

        If the configuration has not been loaded yet, it will be loaded using the default configuration
        based on the model name.

        Returns:
            The configuration as a dict object.
        """
        if self._config is None:
            self._config = self._load_default_config(self.model_name)
        return self._config

    def _configure_task_type(
        self,
        data_root: str | Path | None,
        task: str | None = None,
    ) -> OTXTaskType | None:
        if task is not None:
            return OTXTaskType(task)
        if data_root is None:
            return None

        return configure_task(data_root)

    def _load_default_config(self, model_name: str | None = None) -> dict:
        config_file = DEFAULT_CONFIG_PER_TASK[self.task]
        if model_name is not None:
            model_path = str(config_file).split("/")
            model_path[-1] = f"{model_name}.yaml"
            config_file = Path("/".join(model_path))
        from otx.cli.utils.jsonargparse import get_configuration
        return get_configuration(config_file)

    def get_datamodule(self) -> OTXDataModule:
        """Returns an instance of OTXDataModule with the configured data root.

        Returns:
            OTXDataModule: An instance of OTXDataModule.
        """
        self.config["data"]["config"]["data_root"] = self.data_root
        data_config = self.config["data"]["config"].copy()
        return OTXDataModule(
            task=self.config["data"]["task"],
            config=DataModuleConfig(
                train_subset=SubsetConfig(**data_config.pop("train_subset")),
                val_subset=SubsetConfig(**data_config.pop("val_subset")),
                test_subset=SubsetConfig(**data_config.pop("test_subset")),
                **data_config,
            ),
        )

    def get_model(self, model_name: str | None = None, num_classes: int | None = None) -> OTXModel:
        """Retrieves an instance of the OTXModel class based on the provided model name and number of classes.

        Args:
            model_name (str | None): The name of the model to retrieve. If None, the default model will be used.
            num_classes (int | None): The number of classes for the model.

        Returns:
            OTXModel: An instance of the OTXModel class.

        """
        if model_name is not None:
            self._config = self._load_default_config(self.model_name)
        if num_classes is not None:
            # Add background class for semantic segmentation: Need to check
            if self.task == OTXTaskType.SEMANTIC_SEGMENTATION:
                num_classes += 1
            self.config["model"]["init_args"]["num_classes"] = num_classes
        return instantiate_class(args=(), init=self.config["model"])

    def get_optimizer(self) -> OptimizerCallable:
        """Returns the optimizer callable based on the configuration.

        Returns:
            OptimizerCallable: The optimizer callable.
        """
        return partial_instantiate_class(init=self.config["optimizer"])

    def get_scheduler(self) -> LRSchedulerCallable:
        """Returns the instantiated scheduler based on the configuration.

        Returns:
            LRSchedulerCallable: The instantiated scheduler.
        """
        return partial_instantiate_class(init=self.config["scheduler"])
