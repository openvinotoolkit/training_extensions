# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest
from otx.core.data.dataset.base import LabelInfo
from otx.core.data.module import OTXDataModule
from otx.core.model.entity.base import OTXModel
from otx.core.types.task import OTXTaskType
from otx.engine.utils.auto_configurator import (
    DEFAULT_CONFIG_PER_TASK,
    AutoConfigurator,
    configure_task,
)


@pytest.fixture()
def fxt_data_root_per_task_type() -> dict:
    return {
        "tests/assets/classification_dataset": OTXTaskType.MULTI_CLASS_CLS,
        "tests/assets/multilabel_classification": OTXTaskType.MULTI_LABEL_CLS,
        "tests/assets/car_tree_bug": OTXTaskType.DETECTION,
        "tests/assets/common_semantic_segmentation_dataset": OTXTaskType.SEMANTIC_SEGMENTATION,
        "tests/assets/action_detection_dataset": OTXTaskType.ACTION_DETECTION,
    }


def test_configure_task_with_supported_data_format(fxt_data_root_per_task_type: dict) -> None:
    # Test the configure_task function with a supported data format
    for data_root in fxt_data_root_per_task_type:
        task = configure_task(data_root)
        assert task is not None
        assert isinstance(task, OTXTaskType)
        assert task == fxt_data_root_per_task_type[data_root]


def test_configure_task_with_unsupported_data_format(tmp_path: Path) -> None:
    # Create a temporary directory for testing
    data_root = tmp_path / "data"
    data_root.mkdir()

    # Test the configure_task function with an unsupported data format
    with pytest.raises(ValueError, match="Can't find proper task."):
        configure_task(data_root)


class TestAutoConfigurator:
    def test_check_task(self) -> None:
        # None inputs
        auto_configurator = AutoConfigurator(data_root=None, task=None)
        with pytest.raises(RuntimeError):
            _ = auto_configurator.task

        # data_root is None & task is not None
        auto_configurator = AutoConfigurator(data_root=None, task="MULTI_CLASS_CLS")
        assert auto_configurator.task == "MULTI_CLASS_CLS"

        # data_root is not None & task is None
        data_root = "tests/assets/classification_dataset"
        auto_configurator = AutoConfigurator(data_root=data_root)
        assert auto_configurator.task == "MULTI_CLASS_CLS"

    def test_load_default_config(self) -> None:
        # Test the load_default_config function
        data_root = "tests/assets/classification_dataset"
        task = OTXTaskType.MULTI_CLASS_CLS
        auto_configurator = AutoConfigurator(data_root=data_root, task=task)

        # Default Config
        default_config = auto_configurator._load_default_config()
        target_config = DEFAULT_CONFIG_PER_TASK[task].resolve()
        assert isinstance(default_config, dict)
        assert len(default_config) > 0
        assert "config" in default_config
        assert len(default_config["config"]) > 0
        assert default_config["config"][0] == target_config

        # OTX-Mobilenet-v2
        # new_config
        model_name = "otx_mobilenet_v3_large"
        new_config = auto_configurator._load_default_config(model_name=model_name)
        new_path = str(target_config).split("/")
        new_path[-1] = f"{model_name}.yaml"
        new_target_config = Path("/".join(new_path))
        assert isinstance(new_config, dict)
        assert len(new_config) > 0
        assert "config" in new_config
        assert len(new_config["config"]) > 0
        assert new_config["config"][0] == new_target_config

    def test_get_datamodule(self) -> None:
        data_root = None
        task = OTXTaskType.DETECTION
        auto_configurator = AutoConfigurator(data_root=data_root, task=task)

        # data_root is None
        assert auto_configurator.get_datamodule() is None

        data_root = "tests/assets/car_tree_bug"
        auto_configurator = AutoConfigurator(data_root=data_root, task=task)

        datamodule = auto_configurator.get_datamodule()
        assert isinstance(datamodule, OTXDataModule)
        assert datamodule.task == task

    def test_get_model(self) -> None:
        task = OTXTaskType.DETECTION
        auto_configurator = AutoConfigurator(task=task)

        # Default Model
        model = auto_configurator.get_model()
        assert isinstance(model, OTXModel)
        assert model.num_classes == 1000

        # With meta_info
        label_names = ["class1", "class2", "class3"]
        meta_info = LabelInfo(label_names=label_names, label_groups=[label_names])
        model = auto_configurator.get_model(meta_info=meta_info)
        assert isinstance(model, OTXModel)
        assert model.num_classes == 3

    def test_get_optimizer(self) -> None:
        task = OTXTaskType.SEMANTIC_SEGMENTATION
        auto_configurator = AutoConfigurator(task=task)
        optimizer = auto_configurator.get_optimizer()
        if isinstance(optimizer, list):
            for opt in optimizer:
                assert callable(opt)
        else:
            assert callable(optimizer)

    def test_get_scheduler(self) -> None:
        task = OTXTaskType.INSTANCE_SEGMENTATION
        auto_configurator = AutoConfigurator(task=task)
        scheduler = auto_configurator.get_scheduler()
        if isinstance(scheduler, list):
            for sch in scheduler:
                assert callable(sch)
        else:
            assert callable(scheduler)
