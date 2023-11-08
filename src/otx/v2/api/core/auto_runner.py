"""OTX Core Auto-Runner API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml

from otx.v2.api.core.engine import Engine
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils import set_tuple_constructor
from otx.v2.api.utils.auto_utils import configure_task_type, configure_train_type
from otx.v2.api.utils.decorators import add_subset_dataloader
from otx.v2.api.utils.importing import get_impl_class, get_otx_root_path
from otx.v2.api.utils.type_utils import str_to_task_type, str_to_train_type

ADAPTERS_ROOT = "otx.v2.adapters"
CONFIG_ROOT = f"{get_otx_root_path()}/v2/configs"
DEFAULT_FRAMEWORK_PER_TASK_TYPE: dict[TaskType, dict[str, str | dict]] = {
    TaskType.CLASSIFICATION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.mmengine.mmpretrain",
        "default_config": {TrainType.Incremental: f"{CONFIG_ROOT}/classification/otx_mmpretrain_default.yaml"},
    },
    TaskType.DETECTION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.mmengine.mmdet",
        "default_config": {},
    },
    TaskType.INSTANCE_SEGMENTATION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.mmengine.mmdet",
        "default_config": {},
    },
    TaskType.ROTATED_DETECTION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.mmengine.mmdet",
        "default_config": {},
    },
    TaskType.SEGMENTATION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.mmengine.mmseg",
        "default_config": {},
    },
    TaskType.ACTION_CLASSIFICATION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.mmengine.mmaction",
        "default_config": {TrainType.Incremental: f"{CONFIG_ROOT}/action_classification/otx_mmaction_classification_default.yaml"},
    },
    TaskType.ACTION_DETECTION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.mmengine.mmaction",
        "default_config": {},
    },
    TaskType.ANOMALY_CLASSIFICATION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.lightning.anomalib",
        "default_config": {
            TrainType.Incremental: f"{CONFIG_ROOT}/lightning/otx_anomaly_classification_default.yaml",
        },
    },
    TaskType.ANOMALY_DETECTION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.lightning.anomalib",
        "default_config": {},
    },
    TaskType.ANOMALY_SEGMENTATION: {
        "adapter": f"{ADAPTERS_ROOT}.torch.lightning.anomalib",
        "default_config": {},
    },
    TaskType.VISUAL_PROMPTING: {
        "adapter": f"{ADAPTERS_ROOT}.torch.lightning",
        "default_config": {
            TrainType.Incremental: f"{CONFIG_ROOT}/lightning/otx_visual_prompt_default.yaml",
        },
    },
}


ADAPTER_QUICK_LINK = {
    "mmpretrain": "torch.mmengine.mmpretrain",
    "mmdet": "torch.mmengine.mmdet",
    "mmseg": "torch.mmengine.mmseg",
    "anomalib": "torch.lightning.anomalib",
    "mmaction": "torch.mmengine.mmaction",
    "lightning": "torch.lightning",
}


def set_dataset_paths(config: dict, args: dict) -> dict:
    """Set the dataset paths in the given configuration dictionary based on the provided arguments.

    Args:
        config (dict): The configuration dictionary to update.
        args (dict): The arguments containing the dataset paths.

    Returns:
        dict: The updated configuration dictionary.
    """
    for key, value in args.items():
        if value is None:
            continue
        config["data"][key] = value
    return config


def set_adapters_from_string(
    framework: str,
    engine: bool = True,
    dataset: bool = True,
    get_model: bool = True,
    list_models: bool = True,
    model_configs: bool = True,
) -> dict:
    """Prepare Adapters API class & functions from string.

    Given a framework name, returns a dictionary containing the corresponding engine, dataset builder,
    get_model module, list_models module, and model_configs list. If the framework name is a quick link,
    the corresponding adapter is used. Otherwise, the framework name is used as the adapter.

    Args:
        framework (str): The name of the framework.
        engine (bool): Whether to include the engine in the result. Defaults to True.
        dataset (bool): Whether to include the dataset builder in the result. Defaults to True.
        get_model (bool): Whether to include the get_model module in the result. Defaults to True.
        list_models (bool): Whether to include the list_models module in the result. Defaults to True.
        model_configs (bool): Whether to include the model_configs list in the result. Defaults to True.

    Returns:
        dict: A dictionary containing the corresponding engine, dataset builder, get_model module,
        list_models module, and model_configs list.
    """
    result = {}
    if framework.lower() in ADAPTER_QUICK_LINK:
        adapter = f"{ADAPTERS_ROOT}.{ADAPTER_QUICK_LINK[framework.lower()]}"
    else:
        adapter = framework
    if engine:
        adapter_engine = f"{adapter}.Engine"
        sub_engine = get_impl_class(adapter_engine)
        result["engine"] = sub_engine
    if dataset:
        adapter_dataset = f"{adapter}.Dataset"
        dataset_builder = get_impl_class(adapter_dataset)
        result["dataset"] = dataset_builder
    if get_model:
        get_model_module = get_impl_class(f"{adapter}.get_model")
        result["get_model"] = get_model_module
    if list_models:
        list_models_module = get_impl_class(f"{adapter}.model.list_models")
        result["list_models"] = list_models_module
    if model_configs:
        model_configs_list = get_impl_class(f"{adapter}.model.MODEL_CONFIGS")
        result["model_configs"] = model_configs_list
    return result


@add_subset_dataloader(["train", "val", "test", "unlabeled"])
class AutoRunner:
    """A class for running automated machine learning tasks.

    Example:
        >>> runner = AutoRunner(
            work_dir="output/folder/path",
            train_data_roots="train/data/path",
        )
        >>> runner.train(max_epochs=2)
        {model: Model(), checkpoint: "output/latest/weights.pth"}
    """

    def __init__(
        self,
        *,
        framework: str | None = None,  # ADAPTER_QUICK_LINK.keys()
        task: str | TaskType | None = None,
        train_type: str | TrainType | None = None,
        work_dir: str | None = None,
        train_data_roots: str | None = None,
        train_ann_files: str | None = None,
        val_data_roots: str | None = None,
        val_ann_files: str | None = None,
        test_data_roots: str | None = None,
        test_ann_files: str | None = None,
        unlabeled_data_roots: str | None = None,
        unlabeled_file_list: str | None = None,
        data_format: str | None = None,
        config: dict | str | None = None,
    ) -> None:
        """AutoRunner, which is responsible for OTX's automated training APIs.

        This helps to select the most appropriate type of configuration through auto-detection
            based on framework, task, train_type, and data_roots.

        Args:
            framework (Optional[str], optional): Training frameworks. Refer to otx.v2.adapters. Defaults to None.
            task (Optional[Union[str, TaskType]]): Task of training. Refer to otx.v2.api.entities.task_type.TaskType.
                Defaults to None.
            train_type (Optional[Union[str, TrainType]], optional): Training Type.
                Refer to otx.v2.api.entities.task_type.TrainType. Defaults to None.
            work_dir (Optional[str], optional): Path to the workspace.
                This is where logs and outputs are stored. Defaults to None.
            train_data_roots (Optional[str], optional): The root path of the dataset to be used in the Train subset.
                Defaults to None.
            train_ann_files (Optional[str], optional): The annotation file path of the dataset
                to be used in the Train subset. Defaults to None.
            val_data_roots (Optional[str], optional): The root path of the dataset to be used in the Val subset.
                Defaults to None.
            val_ann_files (Optional[str], optional): The annotation file path of the dataset
                to be used in the validation subset.Defaults to None.
            test_data_roots (Optional[str], optional): The root path of the dataset to be used in the Test subset.
                Defaults to None.
            test_ann_files (Optional[str], optional): The annotation file path of the dataset
                to be used in the test subset. Defaults to None.
            unlabeled_data_roots (Optional[str], optional): The root path of the unlabeled dataset. Defaults to None.
            unlabeled_file_list (Optional[str], optional): The file path containing
                the list of images in the unlabeled dataset. Defaults to None.
            data_format (Optional[str], optional): The format of the dataset want to use. Defaults to None.
            config (Optional[Union[Dict, str]], optional): Path to a configuration file in yaml format.
                Defaults to None.

        Returns:
            None
        """
        self.config_path: str | None = None
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
                _task = dataset_kwargs.get("task")
                task = str_to_task_type(_task) if isinstance(_task, str) else _task
            if train_type is None and dataset_kwargs.get("train_type", None) is not None:
                _train_type = dataset_kwargs.get("train_type")
                train_type = str_to_train_type(_train_type) if isinstance(_train_type, str) else _train_type
        self.auto_configuration(framework, task, train_type)

        self.adapters = set_adapters_from_string(self.framework)
        self.framework_engine = self.adapters.get("engine", None)
        self.dataset_class = self.adapters.get("dataset", None)
        self.get_model = self.adapters.get("get_model", None)
        self.list_models = self.adapters.get("list_models", None)
        self.config_list = self.adapters.get("model_configs", None)

        # Create Dataset Builder
        dataset_kwargs["task"] = self.task
        dataset_kwargs["train_type"] = self.train_type
        dataset_kwargs["data_format"] = self.data_format
        self.dataset = self.dataset_class(**dataset_kwargs)

    def _initial_config(self, config: dict | str | None) -> dict:
        if config is not None:
            if isinstance(config, str):
                self.config_path = config
                set_tuple_constructor()
                with Path(config).open() as file:
                    config = yaml.safe_load(file)
        else:
            config = {}
        if not isinstance(config, dict):
            msg = "Config sould file path of yaml or dictionary"
            raise TypeError(msg)
        if "data" not in config:
            config["data"] = {}
        if "model" not in config:
            config["model"] = {}
        return config

    def configure_model(self, model: str | (dict | (list | object)) | None) -> object:
        """Configure the model to be used for the auto runner.

        Args:
            model (Optional[Union[str, dict, list, object]]): The model to be used for the auto runner.
                Can be a string, dictionary, list, or object.

        Returns:
            object: The configured model object.
        """
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
        framework: str | None,
        task: str | (TaskType | None),
        train_type: str | (TrainType | None),
    ) -> None:
        """Auto-Configuration function for AutoRunner.

        Automatically configures the task type, train type, and framework based on the provided arguments and the
        configuration file. If any of the arguments are None, the method will attempt to retrieve the values from
        the configuration file. If the framework argument is provided, it will be used to set the framework.
        Otherwise, the method will attempt to retrieve the framework from the configuration file.
        If the framework is not found in the configuration file,
        the default framework for the task type will be used.

        Args:
            framework (Optional[str]): The name of the framework to use.
            task (Union[str, TaskType, None]): The task type to use.
            train_type (Union[str, TrainType, None]): The train type to use.

        Returns:
            None
        """
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
        default_setting = DEFAULT_FRAMEWORK_PER_TASK_TYPE[self.task]
        if framework is not None:
            self.framework = framework
        elif not hasattr(self, "framework"):
            if "framework" in self.config:
                self.framework = self.config["framework"]
            elif isinstance(default_setting["adapter"], str):
                self.framework = default_setting["adapter"]
        if self.config_path is None:
            self.config_path = default_setting["default_config"][self.train_type]
            self.config = self._initial_config(self.config_path)

    def build_framework_engine(self) -> None:
        """Create the selected framework Engine."""
        self.engine = self.framework_engine(work_dir=self.work_dir)

    def subset_dataloader(  # noqa: ANN201
        self,
        subset: str,
        pipeline: dict | list | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        **kwargs,
    ):
        """Return a DataLoader for a specific subset of the dataset.

        Args:
            subset (str): The name of the subset to load.
            pipeline (Optional[Union[dict, list]], optional): A pipeline of transforms to apply to the data.
                Defaults to None.
            batch_size (Optional[int], optional): The batch size to use. Defaults to None.
            num_workers (Optional[int], optional): The number of worker threads to use for loading the data.
                Defaults to None.
            **kwargs: Additional keyword arguments to pass to the DataLoader constructor.

        Returns:
            DataLoader: A DataLoader for the specified subset of the dataset.
        """
        subset_dl = self.dataset.subset_dataloader(
            subset=subset,
            pipeline=pipeline,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.subset_dls[subset] = subset_dl
        return subset_dl

    def train(
        self,
        model: str | (dict | (list | object)) | None = None,
        train_dataloader: TypeVar | None = None,
        val_dataloader: TypeVar | None = None,
        optimizer: dict | TypeVar | None = None,
        checkpoint: str | Path | None = None,
        max_iters: int | None = None,
        max_epochs: int | None = None,
        distributed: bool | None = None,
        seed: int | None = None,
        deterministic: bool | None = None,
        precision: str | None = None,
        val_interval: int | None = None,
        **kwargs,
    ) -> dict:
        """Train the model using the data loaders and optimizer.

        Args:
            model (Optional[Union[str, dict, list, object]]):  The models available in each framework's Engine.
                Defaults to None.
            train_dataloader (Optional[TypeVar]): The data loader for training data.
            val_dataloader (Optional[TypeVar]): The data loader for validation data.
            optimizer (Optional[Union[dict, TypeVar]]): The optimizer to use for training.
            checkpoint (Optional[Union[str, Path]]): The path to save the model checkpoint.
            max_iters (Optional[int]): The maximum number of iterations to train for.
            max_epochs (Optional[int]): The maximum number of epochs to train for.
            distributed (Optional[bool]): Whether to use distributed training.
            seed (Optional[int]): The random seed to use for training.
            deterministic (Optional[bool]): Whether to use deterministic training.
            precision (Optional[str]): The precision to use for training.
            val_interval (Optional[int]): The interval at which to perform validation.
            **kwargs (Any): This allows to add arguments that can be accepted by the train function
                of each framework engine.

        Returns:
            dict: A dictionary containing the results of the training.
        """
        dataloader_cfg = {
            "batch_size": kwargs.pop("batch_size", None),
            "num_workers": kwargs.pop("num_workers", None),
        }

        # Configure if dataloader is None
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

        model = self.configure_model(model)

        # Configure Engine
        if not hasattr(self, "engine"):
            self.build_framework_engine()
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
        model: str | (dict | (list | object)) | None = None,
        val_dataloader: dict | object | None = None,
        checkpoint: str | Path | None = None,
        precision: str | None = None,
        **kwargs,
    ) -> dict:
        """Validate the given model on the validation dataset using the specified dataloader and checkpoint.

        If no val dataloader is provided, the val dataloader is obtained from the subset dataloaders
        using the "val" subset. If no such subset exists, a new dataloader is created using the
        "val" subset configuration from the current instance's configuration.

        Args:
            model (Optional[Union[str, dict, list, object]], optional): The model to validate. Defaults to None.
            val_dataloader (Optional[Union[dict, object]], optional): The dataloader to use for validation.
                Defaults to None.
            checkpoint (Optional[Union[str, Path]], optional): The checkpoint to use for validation. Defaults to None.
            precision (Optional[str], optional): The precision to use for validation. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the validation engine.

        Returns:
            dict: A dictionary containing the validation results.
        """
        if val_dataloader is None:
            if self.subset_dls.get("val", None) is not None:
                val_dataloader = self.subset_dls["val"]
            else:
                val_dataloader = self.subset_dataloader(subset="val", config=self.config)

        model = self.configure_model(model)
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
        model: str | (dict | (list | object)) | None = None,
        test_dataloader: dict | object | None = None,
        checkpoint: str | Path | None = None,
        precision: str | None = None,
        **kwargs,
    ) -> dict:
        """Test the given model on the test dataset using the provided test dataloader.

        If no test dataloader is provided, the test dataloader is obtained from the subset dataloaders
        using the "test" subset. If no such subset exists, a new dataloader is created using the
        "test" subset configuration from the current instance's configuration.

        Args:
            model (Optional[Union[str, dict, list, object]]): The model to test. If None, the model
                is obtained from the current instance's configuration.
            test_dataloader (Optional[Union[dict, object]]): The dataloader to use for testing.
                If None, the dataloader is obtained from the subset dataloaders using the "test" subset.
            checkpoint (Optional[Union[str, Path]]): The path to the checkpoint to use for testing.
                If None, the checkpoint is obtained from the current instance's cache.
            precision (Optional[str]): The precision to use for testing. If None, the default precision
                is used.
            **kwargs: Additional keyword arguments to pass to the engine's test method.

        Returns:
            dict: A dictionary containing the test results.
        """
        if test_dataloader is None:
            if self.subset_dls.get("test", None) is not None:
                test_dataloader = self.subset_dls["test"]
            else:
                test_dataloader = self.subset_dataloader(subset="test", config=self.config)
        model = self.configure_model(model)
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
        img: str | (Path | object) | None,
        model: str | (dict | (list | object)) | None = None,
        checkpoint: str | Path | None = None,
        **kwargs,
    ) -> list:
        """Predict the output of the given image using the specified model and checkpoint.

        Args:
            img (Optional[Union[str, Path, object]]): The image to predict the output for.
            model (Optional[Union[str, dict, list, object]]): The model to use for prediction.
            checkpoint (Optional[Union[str, Path]]): The checkpoint to use for prediction.
            **kwargs: Additional keyword arguments to pass to the prediction engine.

        Returns:
            list: A list of predicted outputs for the given image.
        """
        model = self.configure_model(model)
        if checkpoint is None and self.cache.get("checkpoint") is not None:
            checkpoint = self.cache.get("checkpoint")

        # Configure Engine
        if not hasattr(self, "engine"):
            self.build_framework_engine()

        return self.engine.predict(
            model=model,
            img=img,
            checkpoint=checkpoint,
            **kwargs,
        )

    def export(
        self,
        model: str | (dict | (list | object)) | None = None,
        checkpoint: str | Path | None = None,
        precision: str = "float32",
        **kwargs,
    ) -> dict:
        """Export the model to a specified checkpoint.

        Args:
            model (Optional[Union[str, dict, list, object]], optional): The model to export. Defaults to None.
            checkpoint (Optional[Union[str, Path]], optional): The checkpoint to export the model to.
                Defaults to None.
            precision (str, optional): The precision of the exported model. Defaults to "float32".
            **kwargs: Additional keyword arguments to pass to the engine's export method.

        Returns:
            dict: A dictionary containing information about the exported model.
        """
        model = self.configure_model(model)
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
