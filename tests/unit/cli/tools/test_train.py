import argparse

import pytest

from otx.cli.tools import train as target_package
from otx.cli.tools.train import get_args, main
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_args(mocker):
    mock_options = {
        "--train-data-roots": "train/data/root",
        "--val-data-roots": "val/data/root",
        "--unlabeled-data-roots": "unlabeled/data/root",
        "--unlabeled-file-list": "unlabeled/file/list",
        "--load-weights": "weight/path",
        "--resume-from": "resume/path",
        "--save-model-to": "save/path",
        "--work-dir": "work/dir/path",
        "--hpo-time-ratio": "2",
        "--gpus": "0,1",
        "--rdzv-endpoint": "localhost:1",
        "--base-rank": "1",
        "--world-size": "1",
        "--data": "data/yaml",
    }
    mock_command = ["otx"]
    for key, value in mock_options.items():
        mock_command.extend([key, value])

    mocker.patch("sys.argv", mock_command)
    mocker.patch.object(
        target_package, "get_parser_and_hprams_data", return_value=[argparse.ArgumentParser(), {"param": "test"}, []]
    )
    mocker.patch.object(target_package, "add_hyper_parameters_sub_parser", return_value=argparse.ArgumentParser())

    parsed_args, _ = get_args()

    assert parsed_args.train_data_roots == "train/data/root"
    assert parsed_args.val_data_roots == "val/data/root"
    assert parsed_args.unlabeled_data_roots == "unlabeled/data/root"
    assert parsed_args.unlabeled_file_list == "unlabeled/file/list"
    assert parsed_args.load_weights == "weight/path"
    assert parsed_args.resume_from == "resume/path"
    assert parsed_args.save_model_to == "save/path"
    assert parsed_args.work_dir == "work/dir/path"
    assert parsed_args.hpo_time_ratio == 2.0
    assert parsed_args.gpus == "0,1"
    assert parsed_args.rdzv_endpoint == "localhost:1"
    assert parsed_args.base_rank == 1
    assert parsed_args.world_size == 1
    assert parsed_args.data == "data/yaml"


@pytest.fixture
def mock_args(mocker, tmp_path):
    mock_args = mocker.MagicMock()
    mock_args.train_data_roots = "fake_train_data_root"
    mock_args.val_data_roots = "fake_val_data_root"
    mock_args.load_weights = "fake_load_weights"
    mock_args.resume_from = None
    mock_args.save_model_to = tmp_path / "models"
    mock_args.work_dir = tmp_path / "work_dir"
    mock_args.enable_hpo = False
    mock_args.hpo_time_ratio = 4
    mock_args.gpus = None
    mock_args.rdzv_endpoint = "localhost:0"
    mock_args.base_rank = 0
    mock_args.world_size = 0
    mock_args.data = None
    mock_args.unlabeled_data_roots = None
    mock_args.unlabeled_file_list = None

    def mock_contains(self, val):
        return val in self.__dict__

    mock_args.__contains__ = mock_contains
    mock_get_args = mocker.patch("otx.cli.tools.train.get_args")
    mock_get_args.return_value = [mock_args, []]

    return mock_args


@pytest.fixture
def mock_config_manager(mocker):
    mock_config_manager = mocker.patch.object(target_package, "ConfigManager")
    mock_template = mocker.MagicMock()
    mock_template.name = "fake_name"
    mock_config_manager.return_value.template = mock_template
    mock_config_manager.return_value.check_workspace.return_value = True
    mock_config_manager.return_value.get_dataset_config.return_value = {}
    mock_config_manager.return_value.get_hyparams_config.return_value = {}

    return mock_config_manager


@pytest.fixture
def mock_dataset_adapter(mocker):
    mock_dataset_adapter = mocker.patch("otx.cli.tools.train.get_dataset_adapter")
    mock_dataset = mocker.MagicMock()
    mock_label_schema = mocker.MagicMock()
    mock_dataset_adapter.return_value.get_otx_dataset.return_value = mock_dataset
    mock_dataset_adapter.return_value.get_label_schema.return_value = mock_label_schema

    return mock_dataset_adapter


@pytest.fixture
def mock_task(mocker):
    mock_task_class = mocker.MagicMock()
    mock_task = mocker.MagicMock()
    mock_task_class.return_value = mock_task
    mocker.patch.object(target_package, "get_impl_class", return_value=mock_task_class)

    return mock_task


@e2e_pytest_unit
def test_main(mocker, mock_args, mock_config_manager, mock_dataset_adapter, mock_task):
    mocker.patch.object(target_package, "read_label_schema")
    mocker.patch.object(target_package, "read_binary")
    mocker.patch.object(
        target_package,
        "run_hpo",
        return_value=mocker.MagicMock(),
    )
    mocker.patch.object(target_package, "save_model_data")

    ret = main()

    assert ret["retcode"] == 0
