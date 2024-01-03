"""Auto-Configurator class & util functions for OTX Auto-Configuration."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from pathlib import Path
from typing import Any

import datumaro
import yaml
from omegaconf import DictConfig

from otx.core.types.task import OTXTaskType
from otx.core.utils import get_otx_root_path

TASK_TYPE_TO_SUPPORTED_FORMAT = {
    OTXTaskType.MULTI_CLASS_CLS: ["imagenet", "datumaro", "imagenet_with_subset_dirs"],
    OTXTaskType.DETECTION: ["coco", "voc", "yolo"],
    OTXTaskType.SEMANTIC_SEGMENTATION: [
        "cityscapes",
        "common_semantic_segmentation",
        "voc",
        "ade20k2017",
        "ade20k2020",
    ],
    OTXTaskType.INSTANCE_SEGMENTATION: ["coco", "voc"],
}

DEFAULT_MODEL = {
    OTXTaskType.MULTI_CLASS_CLS: "otx_efficientnet_b0",
    OTXTaskType.DETECTION: "atss_mobilenetv2",
    OTXTaskType.SEMANTIC_SEGMENTATION: "litehrnet_18",
    OTXTaskType.INSTANCE_SEGMENTATION: "maskrcnn_r50",
}

DEFAULT_DATA = {
    OTXTaskType.MULTI_CLASS_CLS: "multi_class_cls_default",
    OTXTaskType.DETECTION: "detection_atss",
    OTXTaskType.SEMANTIC_SEGMENTATION: "semantic_segmentation_litehrnet",
    OTXTaskType.INSTANCE_SEGMENTATION: "instance_segmentation_maskrcnn",
}


RECIPE_PATH = Path(get_otx_root_path()) / "recipe"


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


def configure_data_format(data_root: str | Path) -> str:
    """Find the format of dataset."""
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
        data_format = data_formats[0] if "imagenet" not in data_formats else "imagenet"
    return data_format


def replace_key(config: dict, key: str, value: Any) -> None:  # noqa: ANN401
    """Recursively replaces the value of in a nested dictionary with the given key-value.

    Args:
        config (dict): The dictionary to be modified.
        key (str): The key want to replace in the nested dictionary
        value (Any): value to update for key

    Returns:
        None
    """
    for k, v in config.items():
        if k == key:
            config[k] = value
        elif isinstance(v, dict):
            replace_key(v, key, value)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    replace_key(item, key, value)


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
        task: OTXTaskType | None = None,
    ) -> None:
        self._data_root = data_root
        self._task = self._configure_task_type(data_root, task)
        self.model_cfg = DictConfig({})

    @property
    def data_root(self) -> str | Path:
        """Returns the root directory for data storage.

        Raises:
            RuntimeError: If the data root is not set.

        Returns:
            str | Path: The root directory for data storage.
        """
        if self._data_root is None:
            msg = "data_root is None"
            raise RuntimeError(msg)
        return self._data_root

    @data_root.setter
    def data_root(self, data_root: str | Path) -> None:
        self._data_root = data_root
        self._task = self._configure_task_type(data_root, self._task)

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

    @task.setter
    def task(self, task: OTXTaskType) -> None:
        self._task = task

    def _configure_task_type(
        self,
        data_root: str | Path | None,
        task: OTXTaskType | None = None,
    ) -> OTXTaskType | None:
        if task is not None:
            return OTXTaskType(task)
        if data_root is None:
            return None

        data_format = configure_data_format(data_root)
        for task_key, data_value in TASK_TYPE_TO_SUPPORTED_FORMAT.items():
            if data_format in data_value:
                return task_key
        msg = f"Can't find proper task. we are not support {data_format} format, yet."
        raise ValueError(msg)

    def load_default_data_config(self) -> dict:
        """Load the default data configuration for the task.

        Returns:
            dict: The default data configuration.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        config_file = RECIPE_PATH / "data" / f"{DEFAULT_DATA[self.task]}.yaml"
        if config_file.exists():
            with Path(config_file).open() as f:
                return yaml.safe_load(f)
        msg = f"{config_file} does not exist."
        raise FileNotFoundError(msg)

    def load_model_configs(self) -> dict[str, str]:
        """Loads the model configurations from the specified directory.

        Returns:
            A dictionary mapping file names to their corresponding file paths.
        """
        model_dir = RECIPE_PATH / "model" / self.task.value
        file_path_dict = {}
        for file_path in model_dir.iterdir():
            if str(file_path).endswith("yaml") and file_path.is_file():
                file_name = file_path.stem
                file_path_dict[file_name] = str(file_path)
        return file_path_dict

    def load_default_model_config(self, model_name: str | None = None) -> dict:
        """Loads the default model configuration.

        Args:
            model_name (str | None): The name of the model to load the configuration for. If None, the default model
                configuration for the task will be loaded.

        Raises:
            FileNotFoundError: If the specified model configuration file does not exist.

        Returns:
            dict
        """
        if model_name is None:
            model_name = DEFAULT_MODEL[self.task]
        model_dict = self.load_model_configs()
        if model_name in model_dict:
            model_cfg_path = model_dict[model_name]
            with Path(model_cfg_path).open() as f:
                config = yaml.safe_load(f)
            self.model_cfg = DictConfig(config.get("model", config))
            return config
        msg = f"{model_name} does not exist."
        raise FileNotFoundError(msg)

    def load_default_engine_config(self) -> dict:
        """Loads the default engine configuration.

        Returns:
            dict
        """
        config_file = RECIPE_PATH / "engine" / f"{self.task.value}.yaml"
        with Path(config_file).open() as f:
            return yaml.safe_load(f)
