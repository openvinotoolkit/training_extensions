"""OTX Core Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, Union

import yaml

from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.auto_utils import configure_task_type, configure_train_type
from otx.v2.api.utils.importing import get_impl_class
from otx.v2.api.utils.type_utils import str_to_task_type, str_to_train_type

# TODO: Need to organize variables and functions here.
ADAPTERS_ROOT = "otx.v2.adapters"
DEFAULT_FRAMEWORK_PER_TASK_TYPE = {
    TaskType.CLASSIFICATION: f"{ADAPTERS_ROOT}.torch.mmengine.mmpretrain",
    TaskType.DETECTION: f"{ADAPTERS_ROOT}.torch.mmengine.mmdet",
    TaskType.INSTANCE_SEGMENTATION: f"{ADAPTERS_ROOT}.torch.mmengine.mmdet",
    TaskType.ROTATED_DETECTION: f"{ADAPTERS_ROOT}.torch.mmengine.mmdet",
    TaskType.SEGMENTATION: f"{ADAPTERS_ROOT}.torch.mmengine.mmseg",
    # TaskType.ACTION_CLASSIFICATION: f"{ADAPTERS_ROOT}.torch.mmcv.mmaction",
    # TaskType.ACTION_DETECTION: f"{ADAPTERS_ROOT}.torch.mmcv.mmaction",
    TaskType.ANOMALY_CLASSIFICATION: f"{ADAPTERS_ROOT}.torch.anomalib",
    TaskType.ANOMALY_DETECTION: f"{ADAPTERS_ROOT}.torch.anomalib",
    TaskType.ANOMALY_SEGMENTATION: f"{ADAPTERS_ROOT}.torch.anomalib",
}


ADAPTER_QUICK_LINK = {
    "mmpretrain": "torch.mmengine.mmpretrain",
    "mmdet": "torch.mmengine.mmdet",
    "mmseg": "torch.mmengine.mmseg",
    "anomalib": "torch.anomalib",
}


def set_dataset_paths(config: Dict[str, Any], args: Dict[str, str]):
    for key, value in args.items():
        if value is None:
            continue
        subset = key.split("_")[0]
        target = "_".join(key.split("_")[1:])
        if subset not in config["data"]:
            config["data"][subset] = {}
        config["data"][subset][target] = value
    return config


def get_dataset_paths(config: Dict[str, Any]):
    train_config = config["data"].get("train", {})
    val_config = config["data"].get("val", {})
    test_config = config["data"].get("test", {})
    unlabeled_config = config["data"].get("unlabeled", {})
    return {
        "train_data_roots": train_config.get("data_roots", None),
        "train_ann_files": train_config.get("ann_files", None),
        "val_data_roots": val_config.get("data_roots", None),
        "val_ann_files": val_config.get("ann_files", None),
        "test_data_roots": test_config.get("data_roots", None),
        "test_ann_files": test_config.get("ann_files", None),
        "unlabeled_data_roots": unlabeled_config.get("data_roots", None),
        "unlabeled_file_list": unlabeled_config.get("file_list", None),
    }


def set_adapters_from_string(framework: str):
    if framework.lower() in ADAPTER_QUICK_LINK:
        adapter = f"{ADAPTERS_ROOT}.{ADAPTER_QUICK_LINK[framework.lower()]}"
    else:
        adapter = framework
    sub_engine = get_impl_class(f"{adapter}.Engine")
    if sub_engine is None:
        raise NotImplementedError(f"{adapter}.Engine")
    dataset_builder = get_impl_class(f"{adapter}.Dataset")
    if dataset_builder is None:
        raise NotImplementedError(f"{adapter}.Dataset")
    model_build_function = get_impl_class(f"{adapter}.get_model")
    return sub_engine, dataset_builder, model_build_function


class Engine:
    def __init__(self) -> None:
        pass

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def validate(self, *args, **kwargs):
        raise NotImplementedError()

    def test(self, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, *args, **kwargs):
        raise NotImplementedError()

    def export(self, *args, **kwargs):
        raise NotImplementedError()


class AutoEngine:
    def __init__(
        self,
        *,
        framework: Optional[str] = None,  # ADAPTER_QUICK_LINK.keys()
        task: Optional[Union[str, TaskType]] = None,
        train_type: Optional[Union[str, TrainType]] = None,
        work_dir: Optional[str] = None,
        train_data_roots: Optional[str] = None,
        train_ann_files: Optional[str] = None,
        val_data_roots: Optional[str] = None,
        val_ann_files: Optional[str] = None,
        test_data_roots: Optional[str] = None,
        test_ann_files: Optional[str] = None,
        unlabeled_data_roots: Optional[str] = None,
        unlabeled_file_list: Optional[str] = None,
        data_format: Optional[str] = None,
        config: Optional[Union[Dict, str]] = None,
    ):
        self.framework, self.task, self.train_type = None, None, None
        self.work_dir = work_dir
        if config is not None:
            if isinstance(config, str):
                config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
            if not isinstance(config, dict):
                raise TypeError("Config sould file path of yaml or dictionary")
        else:
            # Set
            config = {}
        if "data" not in config:
            config["data"] = {}
        if "model" not in config:
            config["model"] = {}
        self.config = set_dataset_paths(
            config,
            {
                "train_data_roots": train_data_roots,
                "train_ann_files": train_ann_files,
                "val_data_roots": val_data_roots,
                "val_ann_files": val_ann_files,
                "test_data_roots": test_data_roots,
                "test_ann_files": test_ann_files,
                "unlabeled_data_roots": unlabeled_data_roots,
                "unlabeled_file_list": unlabeled_file_list,
            },
        )
        self.data_format = data_format

        # Auto-Configuration
        self.auto_configuration(framework, task, train_type)

        engine_module, self.dataset, self.model_build_function = set_adapters_from_string(self.framework)

        # Create Dataset Builder
        dataset_params = get_dataset_paths(self.config)
        self.dataset_obj = self.dataset(
            task=self.task,
            train_type=self.train_type,
            data_format=data_format,
            **dataset_params,
        )
        self.engine = engine_module(work_dir=self.work_dir)
        # TODO: Check config: if self.config["model"] is empty -> model + data pipeline + recipes selection

    def auto_configuration(self, framework, task, train_type):
        if framework is not None:
            self.framework = framework
        elif hasattr(self.config, "framework"):
            self.framework = self.config.framework

        if task is not None:
            self.task = task
        elif "task" in self.config:
            self.task = self.config["task"]
        if train_type is not None:
            self.train_type = train_type
        elif "train_type" in self.config:
            self.train_type = self.config["train_type"]

        train_config = self.config["data"].get("train", {})
        if self.task is None:
            self.task: str = configure_task_type(train_config.get("data_roots", None), self.data_format)
        if self.train_type is None:
            unlabeled_config = self.config["data"].get("unlabeled", {})
            self.train_type: str = configure_train_type(
                train_config.get("data_roots", None), unlabeled_config.get("data_roots", None)
            )
        if isinstance(self.task, str):
            self.task: TaskType = str_to_task_type(self.task)
        if isinstance(self.train_type, str):
            self.train_type: TrainType = str_to_train_type(self.train_type)
        if self.framework is None:
            self.framework = DEFAULT_FRAMEWORK_PER_TASK_TYPE[self.task]

    def train(
        self,
        # Training
        model=None,
        max_epochs: Optional[int] = None,
        train_data_pipeline=None,
        val_data_pipeline=None,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        # Build DataLoader
        train_dataloader = self.dataset_obj.train_dataloader(
            pipeline=train_data_pipeline,
            config=self.config,
            batch_size=batch_size,
        )
        val_dataloader = self.dataset_obj.val_dataloader(
            pipeline=val_data_pipeline,
            config=self.config,
            batch_size=batch_size,
        )

        # Build Model
        if model is None and self.model_build_function is not None:
            # Model Setting
            model = self.model_build_function(config=self.config, num_classes=self.dataset_obj.num_classes)

        # Training
        # TODO: config + args merge for sub-engine
        return self.engine.train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            work_dir=self.work_dir,
            max_epochs=max_epochs,
            **kwargs,
        )
