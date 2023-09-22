"""OTX Core Auto-Runner API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional, TypeVar, Union

import yaml

from otx.v2.api.core.engine import Engine
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.auto_utils import configure_task_type, configure_train_type
from otx.v2.api.utils.decorators import add_subset_dataloader
from otx.v2.api.utils.importing import get_impl_class, get_otx_root_path
from otx.v2.api.utils.type_utils import str_to_task_type, str_to_train_type

# TODO: Need to organize variables and functions here.
ADAPTERS_ROOT = "otx.v2.adapters"
CONFIG_ROOT = f"{get_otx_root_path()}/v2/configs"
DEFAULT_FRAMEWORK_PER_TASK_TYPE = {
    TaskType.CLASSIFICATION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.mmengine.mmpretrain",
        "default_config": f"{CONFIG_ROOT}/classification/otx_mmpretrain_cli.yaml",
    },
    TaskType.DETECTION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.mmengine.mmdet",
        "default_config": f"{CONFIG_ROOT}/detection/",
    },
    TaskType.INSTANCE_SEGMENTATION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.mmengine.mmdet",
        "default_config": f"{CONFIG_ROOT}/instance_segmentation/",
    },
    TaskType.ROTATED_DETECTION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.mmengine.mmdet",
        "default_config": f"{CONFIG_ROOT}/rotated_detection/",
    },
    TaskType.SEGMENTATION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.mmengine.mmseg",
        "default_config": f"{CONFIG_ROOT}/semantic_segmentation/",
    },
    # TaskType.ACTION_CLASSIFICATION: f"{ADAPTERS_ROOT}.torch.mmcv.mmaction",
    # TaskType.ACTION_DETECTION: f"{ADAPTERS_ROOT}.torch.mmcv.mmaction",
    TaskType.ANOMALY_CLASSIFICATION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.anomalib",
        "default_config": f"{CONFIG_ROOT}/anomaly_classification/otx_anomalib_padim.yaml",
    },
    TaskType.ANOMALY_DETECTION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.anomalib",
        "default_config": f"{CONFIG_ROOT}/anomaly_detection/",
    },
    TaskType.ANOMALY_SEGMENTATION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.anomalib",
        "default_config": f"{CONFIG_ROOT}/anomaly_segmentation/",
    },
}


ADAPTER_QUICK_LINK = {
    "mmpretrain": "torch.mmengine.mmpretrain",
    "mmdet": "torch.mmengine.mmdet",
    "mmseg": "torch.mmengine.mmseg",
    "anomalib": "torch.anomalib",
}


def set_dataset_paths(config: dict, args: dict) -> dict:
    for key, value in args.items():
        if value is None:
            continue
        config["data"][key] = value
    return config


def set_adapters_from_string(framework: str) -> tuple:
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
    get_model = get_impl_class(f"{adapter}.get_model")
    model_configs = get_impl_class(f"{adapter}.model.MODEL_CONFIGS")
    list_models = get_impl_class(f"{adapter}.model.list_models")
    return sub_engine, dataset_builder, get_model, model_configs, list_models


@add_subset_dataloader(["train", "val", "test", "unlabeled"])
class AutoRunner:
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
        config: Optional[Union[dict, str]] = None,
    ) -> None:
        r"""AutoRunner, which is responsible for OTX's automated training APIs.

        This helps to select the most appropriate type of configuration through auto-detection based on framework, task, train_type, and data_roots.

        Args:
            framework (Optional[str], optional): Training frameworks. Refer to otx.v2.adapters. Defaults to None.
            task (Optional[Union[str, TaskType]]): Task of training. Refer to otx.v2.api.entities.task_type.TaskType. Defaults to None.
            train_type (Optional[Union[str, TrainType]], optional): Training Type. Refer to otx.v2.api.entities.task_type.TrainType. Defaults to None.
            work_dir (Optional[str], optional): Path to the workspace. This is where logs and outputs are stored. Defaults to None.
            train_data_roots (Optional[str], optional): The root path of the dataset to be used in the Train subset. Defaults to None.
            train_ann_files (Optional[str], optional): The annotation file path of the dataset to be used in the Train subset. Defaults to None.
            val_data_roots (Optional[str], optional): The root path of the dataset to be used in the Val subset. Defaults to None.
            val_ann_files (Optional[str], optional): The annotation file path of the dataset to be used in the Val subset. Defaults to None.
            test_data_roots (Optional[str], optional): The root path of the dataset to be used in the Test subset. Defaults to None.
            test_ann_files (Optional[str], optional): The annotation file path of the dataset to be used in the Test subset. Defaults to None.
            unlabeled_data_roots (Optional[str], optional): The root path of the unlabeled dataset. Defaults to None.
            unlabeled_file_list (Optional[str], optional): The file path containing the list of images in the unlabeled dataset. Defaults to None.
            data_format (Optional[str], optional): The format of the dataset want to use. Defaults to None.
            config (Optional[Union[Dict, str]], optional): Path to a configuration file in yaml format. Defaults to None.

        Raises:
            TypeError: _description_
        """
        self.config_path: Optional[str] = None
        self.config = self._initial_config(config)
        self.engine: Engine
        self.framework: str
        self.task: TaskType
        self.train_type: TrainType
        self.work_dir = work_dir

        self.cache = {"model": None, "checkpoint": None}
        self.subset_dls: dict = {}

        self.config = set_dataset_paths(
            self.config,
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
        dataset_kwargs = self.config.get("data", None)
        if dataset_kwargs is not None:
            if task is None and dataset_kwargs.get("task", None) is not None:
                task = str_to_task_type(dataset_kwargs.get("task"))
            if train_type is None and dataset_kwargs.get("train_type", None) is not None:
                train_type = str_to_train_type(dataset_kwargs.get("train_type"))
        self.auto_configuration(framework, task, train_type)

        (
            self.framework_engine,
            self.dataset_class,
            self.get_model,
            self.config_list,
            self.list_models,
        ) = set_adapters_from_string(self.framework)

        # Create Dataset Builder
        dataset_kwargs["task"] = self.task
        dataset_kwargs["train_type"] = self.train_type
        dataset_kwargs["data_format"] = self.data_format
        self.dataset = self.dataset_class(**dataset_kwargs)

    def _initial_config(self, config: Optional[Union[dict, str]]) -> dict:
        if config is not None:
            if isinstance(config, str):
                self.config_path = config
                config = yaml.load(open(config), Loader=yaml.FullLoader)
        else:
            config = {}
        if not isinstance(config, dict):
            raise TypeError("Config sould file path of yaml or dictionary")
        if "data" not in config:
            config["data"] = {}
        if "model" not in config:
            config["model"] = {}
        return config

    def _configure_model(self, model: Optional[Union[str, dict, list, object]]) -> object:
        # Configure Model if model is None
        if model is None:
            if self.cache.get("model") is not None:
                model = self.cache.get("model")
            elif self.get_model is not None:
                model = self.get_model(model=self.config, num_classes=self.dataset.num_classes)
        elif isinstance(model, (str, dict)):
            model = self.get_model(model=model)
        return model

    def auto_configuration(
        self,
        framework: Optional[str],
        task: Union[str, TaskType, None],
        train_type: Union[str, TrainType, None],
    ) -> None:
        """A function to automatically detect when framework, task, and train_type are None. This uses data_roots.

        Args:
            framework (Optional[str]): Training frameworks. Refer to otx.v2.adapters.
            task (Optional[Union[str, TaskType]]): Task of training. Refer to otx.v2.api.entities.task_type.TaskType.
            train_type (Optional[Union[str, TrainType]]): Training Type. Refer to otx.v2.api.entities.task_type.TrainType.
        """

        if framework is not None:
            self.framework = framework
        elif hasattr(self.config, "framework"):
            self.framework = self.config.framework

        if task is None and "task" in self.config:
            task = self.config["task"]
        if train_type is None and "train_type" in self.config:
            train_type = self.config["train_type"]

        data_config = self.config.get("data", {})
        data_roots = data_config.get("train_data_roots", data_config.get("test_data_roots", None))
        if task is None:
            task, self.data_format = configure_task_type(data_roots, self.data_format)
        if train_type is None:
            train_type = configure_train_type(data_roots, data_config.get("unlabeled_data_roots", None))
        self.task = str_to_task_type(task) if isinstance(task, str) else task
        self.train_type = str_to_train_type(train_type) if isinstance(train_type, str) else train_type
        if not hasattr(self, "framework"):
            self.framework = DEFAULT_FRAMEWORK_PER_TASK_TYPE[self.task]["adapter"]
        if self.config_path is None:
            self.config_path = DEFAULT_FRAMEWORK_PER_TASK_TYPE[self.task]["default_config"]
            self.config = self._initial_config(self.config_path)

    def build_framework_engine(self) -> None:
        """Create the selected framework Engine."""
        self.engine = self.framework_engine(work_dir=self.work_dir, config=self.config)

    def subset_dataloader(  # noqa: ANN201
        self,
        subset: str,
        pipeline: Optional[Union[dict, list]] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        distributed: bool = False,
        **kwargs,
    ):
        subset_dl = self.dataset.subset_dataloader(subset, pipeline, batch_size, num_workers, distributed, **kwargs)
        self.subset_dls[subset] = subset_dl
        return subset_dl

    def train(
        self,
        model: Optional[Union[str, dict, list, object]] = None,
        train_dataloader: Optional[TypeVar] = None,
        val_dataloader: Optional[TypeVar] = None,
        optimizer: Optional[Union[dict, TypeVar]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        max_iters: Optional[int] = None,
        max_epochs: Optional[int] = None,
        distributed: Optional[bool] = None,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
        precision: Optional[str] = None,
        val_interval: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """The Train function in AutoRunner.

        Each model and dataloader is automatically created based on the configuration if None.

        Args:
            model (optional): The models available in each framework's Engine. Defaults to None.
            train_data_pipeline (optional): Training Dataset's pipeline. Defaults to None.
            val_data_pipeline (optional):  Validation Dataset's pipeline. Defaults to None.
            max_epochs (Optional[int], optional): Specifies the maximum epoch of training. Defaults to None.
            max_iters (Optional[int], optional): Specifies the maximum iters of training. Defaults to None.
            batch_size (Optional[int], optional): Specify the batch size for dataloader. Defaults to None.
            seed (Optional[int], optional): The seed to use for training. Defaults to None.
            deterministic (Optional[bool], optional): The deterministic to use for training. Defaults to None.
            precision (Optional[str], optional): The precision to use for training. Defaults to None.
            kwargs: This allows to add arguments that can be accepted by the train function of each framework engine.

        Returns:
            _type_: Outputs of training.
        """
        # TODO: self.config update with new input
        dataloader_cfg = {
            "batch_size": kwargs.pop("batch_size", None),
            "num_workers": kwargs.pop("num_workers", None),
            "distributed": kwargs.pop("distributed", None),
        }

        # Configure if dataloader is None
        # TODO: Need to add more arguments
        if train_dataloader is None:
            if "train" in self.subset_dls:
                train_dataloader = self.subset_dls["train"]
            else:
                train_dataloader = self.subset_dataloader(subset="train", config=self.config, **dataloader_cfg)
        if val_dataloader is None:
            if "val" in self.subset_dls:
                val_dataloader = self.subset_dls["val"]
            else:
                val_dataloader = self.subset_dataloader(subset="val", config=self.config, **dataloader_cfg)

        model = self._configure_model(model)

        # Configure Engine
        if not hasattr(self, "engine"):
            self.build_framework_engine()
        # TODO: config + args merge for sub-engine
        results = self.engine.train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            checkpoint=checkpoint,
            max_iters=max_iters,
            max_epochs=max_epochs,
            distributed=distributed,
            seed=seed,
            deterministic=deterministic,
            precision=precision,
            val_interval=val_interval,
            **kwargs,
        )
        # Model Cache Update (Training only)
        self.cache = results
        return results

    def validate(
        self,
        model: Optional[Union[str, dict, list, object]] = None,
        val_dataloader: Optional[Union[dict, object]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ) -> dict:
        if val_dataloader is None:
            if self.subset_dls.get("val", None) is not None:
                val_dataloader = self.subset_dls["val"]
            else:
                val_dataloader = self.subset_dataloader(subset="val", config=self.config)

        model = self._configure_model(model)
        if checkpoint is None and self.cache.get("checkpoint") is not None:
            checkpoint = self.cache.get("checkpoint")

        # Configure Engine
        if not hasattr(self, "engine"):
            self.build_framework_engine()

        return self.engine.validate(
            model=model,
            val_dataloader=val_dataloader,
            checkpoint=checkpoint,
            precision=precision,
            **kwargs,
        )

    def test(
        self,
        model: Optional[Union[str, dict, list, object]] = None,
        test_dataloader: Optional[Union[dict, object]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ) -> dict:
        if test_dataloader is None:
            if self.subset_dls.get("test", None) is not None:
                test_dataloader = self.subset_dls["test"]
            else:
                test_dataloader = self.subset_dataloader(subset="test", config=self.config)
        model = self._configure_model(model)
        if checkpoint is None and self.cache.get("checkpoint") is not None:
            checkpoint = self.cache.get("checkpoint")

        # Configure Engine
        if not hasattr(self, "engine"):
            self.build_framework_engine()

        return self.engine.test(
            model=model,
            test_dataloader=test_dataloader,
            checkpoint=checkpoint,
            precision=precision,
            **kwargs,
        )

    def predict(
        self,
        img: Optional[Union[str, Path, object]],
        model: Optional[Union[str, dict, list, object]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        pipeline: Optional[list] = None,
        **kwargs,
    ) -> list:
        model = self._configure_model(model)
        if checkpoint is None and self.cache.get("checkpoint") is not None:
            checkpoint = self.cache.get("checkpoint")

        # Configure Engine
        if not hasattr(self, "engine"):
            self.build_framework_engine()

        return self.engine.predict(
            model=model,
            img=img,
            checkpoint=checkpoint,
            pipeline=pipeline,
            **kwargs,
        )

    def export(
        self,
        model: Optional[Union[str, dict, list, object]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: str = "float32",
        **kwargs,
    ) -> dict:
        model = self._configure_model(model)
        if checkpoint is None and self.cache.get("checkpoint") is not None:
            checkpoint = self.cache.get("checkpoint")

        # Configure Engine
        if not hasattr(self, "engine"):
            self.build_framework_engine()

        return self.engine.export(
            model=model,
            checkpoint=checkpoint,
            precision=precision,
            **kwargs,
        )
