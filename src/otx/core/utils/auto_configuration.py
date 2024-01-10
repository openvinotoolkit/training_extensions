"""Auto-Configurator class & util functions for OTX Auto-Configuration."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datumaro
import yaml
from lightning.pytorch.cli import instantiate_class
from omegaconf import DictConfig

from otx.core.data.module import OTXDataModule
from otx.core.model.entity.base import OTXModel
from otx.core.types.task import OTXTaskType
from otx.core.utils import get_otx_root_path
from otx.core.utils.instantiators import partial_instantiate_class

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.types.transformer_libs import TransformLibType

logger = logging.getLogger()

TASK_TYPE_TO_SUPPORTED_FORMAT = {
    OTXTaskType.MULTI_CLASS_CLS: ["imagenet_with_subset_dirs"],
    OTXTaskType.MULTI_LABEL_CLS: ["datumaro"],
    OTXTaskType.DETECTION: ["coco", "voc", "yolo"],
    OTXTaskType.SEMANTIC_SEGMENTATION: [
        "cityscapes",
        "common_semantic_segmentation_with_subset_dirs",
        "voc",
        "ade20k2017",
        "ade20k2020",
    ],
    OTXTaskType.INSTANCE_SEGMENTATION: ["coco", "voc"],
    OTXTaskType.ACTION_CLASSIFICATION: ["kinetics"],
    OTXTaskType.ACTION_DETECTION: ["ava"],
}

DEFAULT_CONFIG_PER_TASK = {
    OTXTaskType.MULTI_CLASS_CLS: {
        "model": "otx_efficientnet_b0",
        "data": "multi_class_cls_base",
        "optimizer": "sgd",
        "scheduler": "reduce_lr_on_plateau",
    },
    OTXTaskType.MULTI_LABEL_CLS: {
        "model": "efficientnet_b0_light",
        "data": "multi_label_cls_base",
        "optimizer": "sgd",
        "scheduler": "reduce_lr_on_plateau",
    },
    OTXTaskType.DETECTION: {
        "model": "atss_mobilenetv2",
        "data": "detection_atss",
        "optimizer": "sgd",
        "scheduler": "reduce_lr_on_plateau",
    },
    OTXTaskType.INSTANCE_SEGMENTATION: {
        "model": "maskrcnn_r50",
        "data": "instance_segmentation_maskrcnn",
        "optimizer": "sgd",
        "scheduler": "reduce_lr_on_plateau",
    },
    OTXTaskType.SEMANTIC_SEGMENTATION: {
        "model": "litehrnet_18",
        "data": "semantic_segmentation_litehrnet",
        "optimizer": "sgd",
        "scheduler": "reduce_lr_on_plateau",
    },
    OTXTaskType.ACTION_CLASSIFICATION: {
        "model": "x3d",
        "data": "action_classification_x3d",
        "optimizer": "sgd",
        "scheduler": "reduce_lr_on_plateau",
    },
    OTXTaskType.ACTION_DETECTION: {
        "model": "x3d_fastrcnn",
        "data": "action_detection_x3d_fastrcnn",
        "optimizer": "sgd",
        "scheduler": "reduce_lr_on_plateau",
    },
}


CONFIG_PATH = Path(get_otx_root_path()) / "configs" / "_base_"


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
        if k == key and value is not None:
            logger.warning(f"Replace {k} with {value}")
            config[k] = value
        elif isinstance(v, dict):
            replace_key(v, key, value)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    replace_key(item, key, value)


def load_model_configs(task: str) -> dict[str, str]:
    """Loads the model configurations from the specified directory.

    Returns:
        A dictionary mapping file names to their corresponding file paths.
    """
    model_dir = CONFIG_PATH / "model" / task
    file_path_dict = {}
    for file_path in model_dir.iterdir():
        if str(file_path).endswith("yaml") and file_path.is_file():
            file_name = file_path.stem
            file_path_dict[file_name] = str(file_path)
    return file_path_dict


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
    ) -> None:
        self.data_root = data_root

        self._task = self._configure_task_type(data_root, task)

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
        task: str | None = None,
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
        config_file = CONFIG_PATH / "data" / f"{DEFAULT_CONFIG_PER_TASK[self.task]['data']}.yaml"
        if config_file.exists():
            with Path(config_file).open() as f:
                return yaml.safe_load(f)
        msg = f"{config_file} does not exist."
        raise FileNotFoundError(msg)

    def load_default_model_config(self, model: str | None = None, num_classes: int | None = None) -> dict:
        """Load the default model configuration for the task.

        Returns:
            dict: The default model configuration.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        if model is None:
            model = DEFAULT_CONFIG_PER_TASK[self.task]["model"]
        model_dict = load_model_configs(self.task.value)
        if model in model_dict:
            model_cfg_path = model_dict[model]
            with Path(model_cfg_path).open() as f:
                model_config = yaml.safe_load(f)
            # Update num_classes
            replace_key(model_config, "num_classes", num_classes)
            return model_config.get("model", model_config)
        msg = f"{model} does not exist."
        raise FileNotFoundError(msg)

    def load_default_engine_config(self) -> dict:
        """Loads the default engine configuration.

        Returns:
            dict
        """
        config_file = CONFIG_PATH / "engine" / f"{self.task.value}.yaml"
        with Path(config_file).open() as f:
            return yaml.safe_load(f)

    def load_default_optimizer_config(self) -> dict:
        """Loads the default optimizer configuration.

        Returns:
            dict
        """
        config_file = CONFIG_PATH / "optimizer" / f"{DEFAULT_CONFIG_PER_TASK[self.task]['optimizer']}.yaml"
        with Path(config_file).open() as f:
            optimizer_cfg = yaml.safe_load(f)
        return optimizer_cfg.get("optimizer", optimizer_cfg)

    def load_default_scheduler_config(self) -> dict:
        """Loads the default scheduler configuration.

        Returns:
            dict
        """
        config_file = CONFIG_PATH / "scheduler" / f"{DEFAULT_CONFIG_PER_TASK[self.task]['scheduler']}.yaml"
        with Path(config_file).open() as f:
            scheduler_cfg = yaml.safe_load(f)
        return scheduler_cfg.get("scheduler", scheduler_cfg)

    def get_model(self, model: str | None = None, num_classes: int | None = None) -> OTXModel:
        """Get the model from the given model name.

        Args:
            model (str | None): The name of the model to load the configuration for. If None, the default model
                configuration for the task will be loaded.

        Returns:
            OTXModel: The instantiated OTXModel object.
        """
        model_config = self.load_default_model_config(model=model, num_classes=num_classes)
        if not isinstance(model_config, (dict, DictConfig)):
            msg = "Please double-check model config."
            raise TypeError(msg)
        logger.warning(f"Set Default Model: {model_config}")
        return instantiate_class(args=(), init=model_config)

    def get_optimizer(self, **kwargs) -> OptimizerCallable:
        """Get the optimizer from the given optimizer name.

        Returns:
            OptimizerCallable: The instantiated optimizer object.
        """
        optimizer_cfg = self.load_default_optimizer_config()
        for key, value in kwargs.items():
            replace_key(optimizer_cfg, key, value)
        logger.warning(f"Set Default Optimizer: {optimizer_cfg}")
        return partial_instantiate_class(init=optimizer_cfg)

    def get_scheduler(self, **kwargs) -> LRSchedulerCallable:
        """Get the scheduler from the given scheduler name.

        Returns:
            LRSchedulerCallable: The instantiated scheduler object.
        """
        scheduler_cfg = self.load_default_scheduler_config()
        for key, value in kwargs.items():
            replace_key(scheduler_cfg, key, value)
        logger.warning(f"Set Default Scheduler: {scheduler_cfg}")
        return partial_instantiate_class(init=scheduler_cfg)

    def get_datamodule(
        self,
        data_root: str | Path | None = None,
        data_format: str | None = None,
        mem_cache_size: str | None = None,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        include_polygons: bool | None = None,
        batch_size: int | dict[str, int] | None = None,
        num_workers: int | dict[str, int] | None = None,
        transforms: list | dict[str, list] | None = None,
        transform_lib_type: TransformLibType | dict[str, TransformLibType] | None = None,
    ) -> OTXDataModule:
        """Returns an instance of OTXDataModule based on the provided configuration.

        Args:
            data_root (str): The root directory of the data.
            task (OTXTaskType): The type of OTX task.

            data_format (str | None, optional): The format of the data. Defaults to None.
            mem_cache_size (str | None, optional): The size of the memory cache. Defaults to None.
            mem_cache_img_max_size (tuple[int, int] | None, optional): The maximum size of images in the memory cache.
                Defaults to None.
            include_polygons (bool | None, optional): Whether to include polygons. Defaults to None.

            batch_size (int | dict[str, int] | None, optional): The batch size. Defaults to None.
            num_workers (int | dict[str, int] | None, optional): The number of workers. Defaults to None.
            transforms (list | dict[str, list] | None, optional): The list of transforms. Defaults to None.
            transform_lib_type (TransformLibType | dict[str, TransformLibType] | None, optional):
                The type of transform library. Defaults to None.

        Returns:
            OTXDataModule: An instance of OTXDataModule based on the provided configuration.
        """
        # Load default DataModuleConfig
        if data_root is None:
            data_root = self.data_root
        default_config = self.load_default_data_config()

        # Update DataModuleConfig
        default_config["config"]["data_root"] = data_root
        datamodule_arguments = {
            "data_format": data_format,
            "mem_cache_size": mem_cache_size,
            "mem_cache_img_max_size": mem_cache_img_max_size,
            "include_polygons": include_polygons,
        }
        for datamodule_key, datamodule_value in datamodule_arguments.items():
            if datamodule_value is not None:
                default_config["config"][datamodule_key] = datamodule_value

        # Update SubsetConfig
        subset_arguments = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "transforms": transforms,
            "transform_lib_type": transform_lib_type,
        }
        for subset_key, subset_value in subset_arguments.items():
            if subset_value is not None:
                if isinstance(subset_value, dict):
                    for subset_name, value in subset_value.items():
                        default_config["config"][subset_name][subset_key] = value
                else:
                    default_config["config"]["train_subset"][subset_key] = subset_value
                    default_config["config"]["val_subset"][subset_key] = subset_value
                    default_config["config"]["test_subset"][subset_key] = subset_value

        # Return OTXDataModule
        self._datamodule = OTXDataModule.from_config(config=default_config)
        return self._datamodule
