"""OTX V2 API-core Unit-Test codes (AutoRunner)."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.v2.api.core.auto_runner import AutoRunner, set_adapters_from_string, set_dataset_paths
from otx.v2.api.entities.task_type import TaskType, TrainType
from pytest_mock.plugin import MockerFixture


class TestAutoRunner:
    @pytest.fixture()
    def auto_runner(self, mocker: MockerFixture) -> AutoRunner:
        """Fixture that returns an instance of AutoRunner."""
        mocker.patch("otx.v2.api.core.auto_runner.configure_task_type", return_value=("classification", "imagenet"))
        mocker.patch("otx.v2.api.core.auto_runner.configure_train_type", return_value="Incremental")

        return AutoRunner()

    def test_init(self, auto_runner: AutoRunner, mocker: MockerFixture) -> None:
        """Test the initialization of AutoRunner."""
        assert isinstance(auto_runner, AutoRunner)

        auto_runner = AutoRunner(task="anomaly_classification")
        assert auto_runner.framework == "otx.v2.adapters.torch.anomalib"
        assert auto_runner.task == TaskType.ANOMALY_CLASSIFICATION

        with pytest.raises(TypeError):
            AutoRunner(config=mocker.MagicMock())

    def test_init_from_config(self, mocker: MockerFixture) -> None:
        """Test the initialization of AutoRunner from a configuration."""
        config = {"framework": "mmpretrain", "task": "classification", "train_type": "Incremental"}
        auto_runner = AutoRunner(config=config)
        assert auto_runner.framework == "mmpretrain"
        assert auto_runner.task == TaskType.CLASSIFICATION
        assert auto_runner.train_type == TrainType.Incremental

        config_2 = {"framework": "mmpretrain", "data": {"task": "classification", "train_type": "Incremental"}}
        auto_runner = AutoRunner(config=config_2)
        assert auto_runner.framework == "mmpretrain"
        assert auto_runner.task == TaskType.CLASSIFICATION
        assert auto_runner.train_type == TrainType.Incremental

        config_path = "/path/configs.yaml"
        mocker.patch("otx.v2.api.core.auto_runner.Path.open")
        mocker.patch("otx.v2.api.core.auto_runner.yaml.safe_load", return_value=config)
        auto_runner = AutoRunner(config=config_path)
        assert auto_runner.framework == "mmpretrain"
        assert auto_runner.task == TaskType.CLASSIFICATION
        assert auto_runner.train_type == TrainType.Incremental


    def test_auto_configuration(self, auto_runner: AutoRunner) -> None:
        """Test the auto configuration of AutoRunner."""
        auto_runner.auto_configuration(
            framework="anomalib",
            task=None,
            train_type=None,
        )
        assert auto_runner.framework == "anomalib"

        auto_runner.auto_configuration(
            framework="anomalib",
            task="segmentation",
            train_type=None,
        )
        assert auto_runner.task == TaskType.SEGMENTATION

        auto_runner.auto_configuration(
            framework="anomalib",
            task="segmentation",
            train_type="Semisupervised",
        )
        assert auto_runner.train_type == TrainType.Semisupervised

    def test_configure_model(self, auto_runner: AutoRunner, mocker: MockerFixture) -> None:
        """Test the configuration of the model in AutoRunner."""
        mock_model = mocker.MagicMock()
        auto_runner.dataset = mocker.MagicMock()
        auto_runner.get_model = mocker.MagicMock()
        auto_runner.get_model.return_value = mock_model
        model = auto_runner.configure_model(model=None)
        assert model == mock_model

        mock_model_2 = mocker.MagicMock()
        auto_runner.cache = {"model": mock_model_2, "checkpoint": "path/ckpt"}
        model = auto_runner.configure_model(model=None)
        assert model == mock_model_2

        model = auto_runner.configure_model(model="model_name")
        assert model == mock_model

    def test_build_framework_engine(self, auto_runner: AutoRunner) -> None:
        """Test the building of the framework engine in AutoRunner."""
        auto_runner.build_framework_engine()
        assert auto_runner.engine.__class__.__name__ == "MMPTEngine"

    def test_subset_dataloader(self, auto_runner: AutoRunner, mocker: MockerFixture) -> None:
        """Test the creation of a subset dataloader in AutoRunner."""
        auto_runner.dataset = mocker.MagicMock()
        data_loader = auto_runner.subset_dataloader("train")
        assert auto_runner.subset_dls["train"] == data_loader

    def test_train(self, auto_runner: AutoRunner, mocker: MockerFixture) -> None:
        """Test the training process in AutoRunner."""
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.subset_dataloader", return_value=None)
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.build_framework_engine", return_value=mocker.MagicMock())
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.configure_model", return_value=mocker.MagicMock())
        # If the framework_engine is not ready, it will raise an AttributeError.
        with pytest.raises(AttributeError):
            auto_runner.train()

        auto_runner.engine = mocker.MagicMock()
        auto_runner.engine.train.return_value = {"model": mocker.MagicMock(), "checkpoint": "path/ckpt"}
        result = auto_runner.train()
        auto_runner.engine.train.assert_called_once()
        assert isinstance(result, dict)
        assert "model" in result
        assert "checkpoint" in result

    def test_train_with_prepared_subset_dl(self, auto_runner: AutoRunner, mocker: MockerFixture) -> None:
        """Test the training process in AutoRunner with a prepared subset dataloader."""
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.build_framework_engine", return_value=mocker.MagicMock())
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.configure_model", return_value=mocker.MagicMock())
        auto_runner.dataset = mocker.MagicMock()
        train_data_loader = auto_runner.subset_dataloader("train")
        assert auto_runner.subset_dls["train"] == train_data_loader
        val_data_loader = auto_runner.subset_dataloader("val")
        assert auto_runner.subset_dls["val"] == val_data_loader

        auto_runner.engine = mocker.MagicMock()
        auto_runner.engine.train.return_value = {"model": mocker.MagicMock(), "checkpoint": "path/ckpt"}
        result = auto_runner.train()
        auto_runner.engine.train.assert_called_once()
        assert isinstance(result, dict)
        assert "model" in result
        assert "checkpoint" in result

    def test_validate(self, auto_runner: AutoRunner, mocker: MockerFixture) -> None:
        """Test the validation process in AutoRunner."""
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.subset_dataloader", return_value=None)
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.build_framework_engine", return_value=mocker.MagicMock())
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.configure_model", return_value=mocker.MagicMock())
        # If the framework_engine is not ready, it will raise an AttributeError.
        with pytest.raises(AttributeError):
            auto_runner.validate()

        # Normal Case
        auto_runner.engine = mocker.MagicMock()
        auto_runner.engine.validate.return_value = {"accuracy": 0.9}
        result = auto_runner.validate()
        auto_runner.engine.validate.assert_called_once()
        assert isinstance(result, dict)
        assert "accuracy" in result

        # Case with checkpoint cache
        auto_runner.cache = {"model": mocker.MagicMock(), "checkpoint": "path/ckpt"}
        auto_runner.validate()
        assert "checkpoint" in auto_runner.engine.validate.call_args_list[-1][-1]
        assert auto_runner.engine.validate.call_args_list[-1][-1]["checkpoint"] == "path/ckpt"

    def test_validate_with_prepared_dl(self, auto_runner: AutoRunner, mocker: MockerFixture) -> None:
        """Test the validation process in AutoRunner with a prepared subset dataloader."""
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.build_framework_engine", return_value=mocker.MagicMock())
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.configure_model", return_value=mocker.MagicMock())

        auto_runner.dataset = mocker.MagicMock()
        val_data_loader = auto_runner.subset_dataloader("val")
        assert auto_runner.subset_dls["val"] == val_data_loader

        auto_runner.engine = mocker.MagicMock()
        auto_runner.engine.validate.return_value = {"accuracy": 0.9}
        result = auto_runner.validate()
        auto_runner.engine.validate.assert_called_once()
        assert isinstance(result, dict)
        assert "accuracy" in result

    def test_test(self, auto_runner: AutoRunner, mocker: MockerFixture) -> None:
        """Test the testing process in AutoRunner."""
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.subset_dataloader", return_value=None)
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.build_framework_engine", return_value=mocker.MagicMock())
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.configure_model", return_value=mocker.MagicMock())
        # If the framework_engine is not ready, it will raise an AttributeError.
        with pytest.raises(AttributeError):
            auto_runner.test()

        # Normal Case
        auto_runner.engine = mocker.MagicMock()
        auto_runner.engine.test.return_value = {"accuracy": 0.9}
        result = auto_runner.test()
        auto_runner.engine.test.assert_called_once()
        assert isinstance(result, dict)
        assert "accuracy" in result

        # Case with checkpoint cache
        auto_runner.cache = {"model": mocker.MagicMock(), "checkpoint": "path/ckpt"}
        auto_runner.test()
        assert "checkpoint" in auto_runner.engine.test.call_args_list[-1][-1]
        assert auto_runner.engine.test.call_args_list[-1][-1]["checkpoint"] == "path/ckpt"

    def test_test_with_prepared_dl(self, auto_runner: AutoRunner, mocker: MockerFixture) -> None:
        """Test the testing process in AutoRunner with a prepared subset dataloader."""
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.build_framework_engine", return_value=mocker.MagicMock())
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.configure_model", return_value=mocker.MagicMock())

        auto_runner.dataset = mocker.MagicMock()
        test_data_loader = auto_runner.subset_dataloader("test")
        assert auto_runner.subset_dls["test"] == test_data_loader

        auto_runner.engine = mocker.MagicMock()
        auto_runner.engine.test.return_value = {"accuracy": 0.9}
        result = auto_runner.test()
        auto_runner.engine.test.assert_called_once()
        assert isinstance(result, dict)
        assert "accuracy" in result

    def test_predict(self, auto_runner: AutoRunner, mocker: MockerFixture) -> None:
        """Test the prediction process in AutoRunner."""
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.build_framework_engine", return_value=mocker.MagicMock())
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.configure_model", return_value=mocker.MagicMock())
        # If the framework_engine is not ready, it will raise an AttributeError.
        with pytest.raises(AttributeError):
            auto_runner.predict(img=mocker.MagicMock())

        # Normal Case
        auto_runner.engine = mocker.MagicMock()
        auto_runner.engine.predict.return_value = {"pred_score": 0.9}
        result = auto_runner.predict(img=mocker.MagicMock())
        auto_runner.engine.predict.assert_called_once()
        assert isinstance(result, dict)
        assert "pred_score" in result

        # Case with checkpoint cache
        auto_runner.cache = {"model": mocker.MagicMock(), "checkpoint": "path/ckpt"}
        auto_runner.predict(img=mocker.MagicMock())
        assert "checkpoint" in auto_runner.engine.predict.call_args_list[-1][-1]
        assert auto_runner.engine.predict.call_args_list[-1][-1]["checkpoint"] == "path/ckpt"

    def test_export(self, auto_runner: AutoRunner, mocker: MockerFixture) -> None:
        """Test the exporting process in AutoRunner."""
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.build_framework_engine", return_value=mocker.MagicMock())
        mocker.patch("otx.v2.api.core.auto_runner.AutoRunner.configure_model", return_value=mocker.MagicMock())
        # If the framework_engine is not ready, it will raise an AttributeError.
        with pytest.raises(AttributeError):
            auto_runner.export()

        # Normal Case
        auto_runner.engine = mocker.MagicMock()
        auto_runner.engine.export.return_value = {"openvino": "ir/path"}
        result = auto_runner.export()
        auto_runner.engine.export.assert_called_once()
        assert isinstance(result, dict)
        assert "openvino" in result

        # Case with checkpoint cache
        auto_runner.cache = {"model": mocker.MagicMock(), "checkpoint": "path/ckpt"}
        auto_runner.export()
        assert "checkpoint" in auto_runner.engine.export.call_args_list[-1][-1]
        assert auto_runner.engine.export.call_args_list[-1][-1]["checkpoint"] == "path/ckpt"


def test_set_dataset_paths() -> None:
    """
    Test function for set_dataset_paths().

    Steps:
    1. Test with All arguments are None.
    2. Test with Some arguments are None.
    3. Test with All arguments are not None.
    """
    # Test case 1: All arguments are None
    config = {"data": {"train": None, "val": None}}
    args = {"train": None, "val": None}
    assert set_dataset_paths(config, args) == {"data": {"train": None, "val": None}}

    # Test case 2: Some arguments are None
    config = {"data": {"train": None, "val": None}}
    args = {"train": "/path/to/train", "val": None}
    assert set_dataset_paths(config, args) == {"data": {"train": "/path/to/train", "val": None}}

    # Test case 3: All arguments are not None
    config = {"data": {"train": None, "val": None}}
    args = {"train": "/path/to/train", "val": "/path/to/val"}
    assert set_dataset_paths(config, args) == {"data": {"train": "/path/to/train", "val": "/path/to/val"}}


def test_set_adapters_from_string() -> None:
    """
    Test function for set_adapters_from_string().

    Steps:
    1. Test with engine=False, dataset=False, get_model=False, list_models=False, model_configs=False.
    2. Test with engine=True, dataset=False, get_model=False, list_models=False, model_configs=False.
    3. Test with engine=False, dataset=True, get_model=False, list_models=False, model_configs=False.
    4. Test with default arguments.
    5. Test with a non-existent framework name.
    """
    framework = "mmpretrain"
    result = set_adapters_from_string(framework=framework, engine=False, dataset=False, get_model=False, list_models=False, model_configs=False)
    assert len(result) == 0
    
    result = set_adapters_from_string(framework=framework, engine=True, dataset=False, get_model=False, list_models=False, model_configs=False)
    assert "engine" in result
    assert result["engine"].__name__ == "MMPTEngine"

    result = set_adapters_from_string(framework=framework, engine=False, dataset=True, get_model=False, list_models=False, model_configs=False)
    assert "dataset" in result
    assert result["dataset"].__name__ == "Dataset"

    result = set_adapters_from_string(framework=framework)
    assert "engine" in result
    assert "dataset" in result
    assert "get_model" in result
    assert "list_models" in result
    assert "model_configs" in result

    with pytest.raises(ModuleNotFoundError):
        set_adapters_from_string(framework="no_named")
