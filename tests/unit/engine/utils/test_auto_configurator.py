# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest
import torch
from otx.core.data.module import OTXDataModule
from otx.core.model.base import OTXModel
from otx.core.types.label import LabelInfo, SegLabelInfo
from otx.core.types.task import OTXTaskType
from otx.core.types.transformer_libs import TransformLibType
from otx.engine.utils import auto_configurator as target_file
from otx.engine.utils.auto_configurator import (
    DEFAULT_CONFIG_PER_TASK,
    AutoConfigurator,
    configure_task,
)
from otx.utils.utils import should_pass_label_info


@pytest.fixture()
def fxt_data_root_per_task_type() -> dict:
    return {
        "tests/assets/classification_dataset": OTXTaskType.MULTI_CLASS_CLS,
        "tests/assets/multilabel_classification": OTXTaskType.MULTI_LABEL_CLS,
        "tests/assets/car_tree_bug": OTXTaskType.DETECTION,
        "tests/assets/common_semantic_segmentation_dataset/supervised": OTXTaskType.SEMANTIC_SEGMENTATION,
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
    (data_root / "1.jpg").open("a").close()  # Dummy image file

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
        model_name = "deit_tiny"
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

    def test_get_datamodule_set_input_size_multiplier(self, mocker) -> None:
        mock_otxdatamodule = mocker.patch.object(target_file, "OTXDataModule")
        auto_configurator = AutoConfigurator(
            data_root="tests/assets/car_tree_bug",
            task=OTXTaskType.DETECTION,
            model_name="yolox_tiny",
        )
        auto_configurator.config["data"]["adaptive_input_size"] = "auto"

        auto_configurator.get_datamodule()

        assert mock_otxdatamodule.call_args.kwargs["input_size_multiplier"] == 32

    def test_get_model(self, fxt_task: OTXTaskType) -> None:
        if fxt_task is OTXTaskType.H_LABEL_CLS:
            pytest.xfail(reason="Not working")

        auto_configurator = AutoConfigurator(task=fxt_task)

        # With label_info
        label_names = ["class1", "class2", "class3"]
        label_info = (
            LabelInfo(label_names=label_names, label_groups=[label_names])
            if fxt_task != OTXTaskType.SEMANTIC_SEGMENTATION
            else SegLabelInfo(label_names=label_names, label_groups=[label_names])
        )
        model = auto_configurator.get_model(label_info=label_info)
        assert isinstance(model, OTXModel)

        model_cls = model.__class__

        if should_pass_label_info(model_cls):
            with pytest.raises(ValueError, match="Given model class (.*) requires a valid label_info to instantiate."):
                _ = auto_configurator.get_model(label_info=None)

    def test_get_model_set_input_size(self) -> None:
        auto_configurator = AutoConfigurator(task=OTXTaskType.MULTI_CLASS_CLS)
        label_names = ["class1", "class2", "class3"]
        label_info = LabelInfo(label_names=label_names, label_groups=[label_names])
        input_size = 300

        model = auto_configurator.get_model(label_info=label_info, input_size=input_size)

        assert model.input_size == (input_size, input_size)

    def test_get_optimizer(self, fxt_task: OTXTaskType) -> None:
        if fxt_task in {
            OTXTaskType.ANOMALY_SEGMENTATION,
            OTXTaskType.ANOMALY_DETECTION,
            OTXTaskType.ANOMALY_CLASSIFICATION,
        }:
            pytest.xfail(reason="Not working")

        auto_configurator = AutoConfigurator(task=fxt_task)
        optimizer = auto_configurator.get_optimizer()
        if isinstance(optimizer, list):
            for opt in optimizer:
                assert callable(opt)
        else:
            assert callable(optimizer)

    def test_get_scheduler(self, fxt_task: OTXTaskType) -> None:
        if fxt_task in {
            OTXTaskType.ANOMALY_SEGMENTATION,
            OTXTaskType.ANOMALY_DETECTION,
            OTXTaskType.ANOMALY_CLASSIFICATION,
        }:
            pytest.xfail(reason="Not working")

        auto_configurator = AutoConfigurator(task=fxt_task)
        scheduler = auto_configurator.get_scheduler()
        if isinstance(scheduler, list):
            for sch in scheduler:
                assert callable(sch)
        else:
            assert callable(scheduler)

    def test_update_ov_subset_pipeline(self) -> None:
        data_root = "tests/assets/car_tree_bug"
        auto_configurator = AutoConfigurator(data_root=data_root, task="DETECTION")

        datamodule = auto_configurator.get_datamodule()
        assert datamodule.test_subset.transforms == [
            {
                "class_path": "otx.core.data.transform_libs.torchvision.Resize",
                "init_args": {
                    "scale": (800, 992),
                    "is_numpy_to_tvtensor": True,
                },
            },
            {"class_path": "torchvision.transforms.v2.ToDtype", "init_args": {"dtype": torch.float32}},
            {
                "class_path": "torchvision.transforms.v2.Normalize",
                "init_args": {"mean": [0.0, 0.0, 0.0], "std": [255.0, 255.0, 255.0]},
            },
        ]

        assert datamodule.test_subset.transform_lib_type == TransformLibType.TORCHVISION

        updated_datamodule = auto_configurator.update_ov_subset_pipeline(datamodule, subset="test")
        assert updated_datamodule.test_subset.transforms == [{"class_path": "torchvision.transforms.v2.ToImage"}]

        assert updated_datamodule.test_subset.transform_lib_type == TransformLibType.TORCHVISION
        assert not updated_datamodule.tile_config.enable_tiler
        assert updated_datamodule.unlabeled_subset.data_root is None
