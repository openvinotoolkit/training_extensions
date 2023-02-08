import yaml
from os import path as osp
from tempfile import TemporaryDirectory

import pytest
from otx.cli.utils.config import override_parameters, configure_dataset

from tests.test_suite.e2e_test_system import e2e_pytest_unit

@pytest.fixture
def mock_parameters():
    return {
        "a" : {"a.a" : {"value" : 1, "default_value" : 2}}
    }

@e2e_pytest_unit
def test_override_mock_parameters(mock_parameters):
    overrides = {
        "a" : {"a.a" : {"value" : 3, "default_value" : 4}}
    }
    override_parameters(overrides, mock_parameters)

    assert mock_parameters["a"]["a.a"]["value"] == 3
    assert mock_parameters["a"]["a.a"]["default_value"] == 4

@e2e_pytest_unit
def test_override_mock_parameters_not_allowed_key(mock_parameters):
    overrides = {
        "a" : {"a.a" : {"fake" : 3, "default_value" : 4}}
    }

    with pytest.raises(ValueError):
        override_parameters(overrides, mock_parameters)

@e2e_pytest_unit
def test_override_mock_parameters_non_exist_key(mock_parameters):
    overrides = {
        "a" : {"a.b" : {"value" : 3, "default_value" : 4}}
    }

    with pytest.raises(ValueError):
        override_parameters(overrides, mock_parameters)

@e2e_pytest_unit
def test_configure_dataset(mocker):
    # prepare
    def mock_contain(self, key):
        return key in self.__dict__
    mock_args = mocker.MagicMock()
    mock_args.__contains__ = mock_contain
    mock_args.train_ann_files = "train_ann_files"
    mock_args.train_data_roots = "train_data_roots"
    mock_args.val_ann_files = "val_ann_files"
    mock_args.val_data_roots = "val_data_roots"
    mock_args.unlabeled_file_list = "unlabeled_file_list"
    mock_args.unlabeled_data_roots = "unlabeled_data_roots"
    mock_args.test_ann_files = "test_ann_files"
    mock_args.test_data_roots = "test_data_roots"
    mock_args.data = None
    # run
    data_config = configure_dataset(mock_args)

    # check
    assert data_config["data"]["train"]["ann-files"] == mock_args.train_ann_files
    assert data_config["data"]["train"]["data-roots"] == mock_args.train_data_roots
    assert data_config["data"]["val"]["ann-files"] == mock_args.val_ann_files
    assert data_config["data"]["val"]["data-roots"] == mock_args.val_data_roots
    assert data_config["data"]["unlabeled"]["file-list"] == mock_args.unlabeled_file_list
    assert data_config["data"]["unlabeled"]["data-roots"] == mock_args.unlabeled_data_roots
    assert data_config["data"]["test"]["ann-files"] == mock_args.test_ann_files
    assert data_config["data"]["test"]["data-roots"] == mock_args.test_data_roots

@e2e_pytest_unit
def test_configure_dataset_with_data_args(mocker):
    mock_args = mocker.MagicMock()

    with TemporaryDirectory() as tmp_dir:
        mock_args.data = osp.join(tmp_dir, "data.yaml")
        mock_data = {"data" : {"train" : {"ann-files" : "a", "data-roots" : "b"}}}
        with open(mock_args.data, "w") as f:
            yaml.dump(mock_data, f)

        data_config = configure_dataset(mock_args)

    assert data_config["data"]["train"]["ann-files"] == mock_data["data"]["train"]["ann-files"]
    assert data_config["data"]["train"]["data-roots"] == mock_data["data"]["train"]["data-roots"]
