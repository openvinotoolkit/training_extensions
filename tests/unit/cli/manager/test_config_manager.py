import argparse
import tempfile

import pytest
from omegaconf import DictConfig, OmegaConf

from otx.cli.manager.config_manager import (
    DEFAULT_MODEL_TEMPLATE_ID,
    ConfigManager,
    set_workspace,
)
from otx.cli.registry import Registry
from otx.cli.utils.errors import (
    CliException,
    ConfigValueError,
    FileNotExistError,
    NotSupportedError,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def test_set_workspace():
    task = "CLASSIFICATION"
    root = "/home/user/data"
    name = "otx-workspace"

    expected_path = f"{root}/{name}-{task}"
    assert set_workspace(task, root, name) == expected_path

    expected_path = f"./{name}-{task}"
    assert set_workspace(task, name=name) == expected_path


class TestConfigManager:
    @pytest.fixture
    def config_manager(self, mocker):
        args = mocker.MagicMock()
        args.template = "."
        args.config_path = "path/to/config.yaml"
        args.workspace_path = "path/to/workspace"
        args.mode = "train"
        args.task_type = "classification"
        args.train_type = "incremental"
        return ConfigManager(args)

    def get_default_template(self, otx_root, task_type):
        otx_registry = Registry(otx_root).filter(task_type=task_type)
        return otx_registry.get(DEFAULT_MODEL_TEMPLATE_ID[task_type.upper()])

    @e2e_pytest_unit
    def test_check_workspace(self, mocker, config_manager):
        mock_exists = mocker.patch("otx.cli.manager.config_manager.Path.exists")
        # Define the return value of the `os.path.exists` function
        mock_exists.return_value = True
        # Call the function and check the returned value
        assert config_manager.check_workspace()
        mock_exists.return_value = False
        assert not config_manager.check_workspace()

    @e2e_pytest_unit
    def test_get_arg_data_yaml(self, mocker):
        # Call the function and check the returned value
        args = mocker.MagicMock()
        args.template = "."
        args.train_data_roots = "path/to/data/train"
        args.val_data_roots = "path/to/data/val"
        args.test_data_roots = "path/to/data/test"
        args.unlabeled_data_roots = None
        args.mode = "train"
        config_manager = ConfigManager(args)
        assert config_manager._get_arg_data_yaml() == {
            "data": {
                "train": {"ann-files": None, "data-roots": "path/to/data/train"},
                "val": {"ann-files": None, "data-roots": "path/to/data/val"},
                "test": {"ann-files": None, "data-roots": None},
                "unlabeled": {"file-list": None, "data-roots": None},
            }
        }
        config_manager.mode = "test"
        assert config_manager._get_arg_data_yaml() == {
            "data": {
                "train": {"ann-files": None, "data-roots": None},
                "val": {"ann-files": None, "data-roots": None},
                "test": {"ann-files": None, "data-roots": "path/to/data/test"},
                "unlabeled": {"file-list": None, "data-roots": None},
            }
        }

        args.unlabeled_data_roots = "path/to/data/unlabeled"
        config_manager = ConfigManager(args)
        assert config_manager._get_arg_data_yaml() == {
            "data": {
                "train": {"ann-files": None, "data-roots": "path/to/data/train"},
                "val": {"ann-files": None, "data-roots": "path/to/data/val"},
                "test": {"ann-files": None, "data-roots": None},
                "unlabeled": {"file-list": None, "data-roots": "path/to/data/unlabeled"},
            }
        }

    @e2e_pytest_unit
    def test_create_empty_data_cfg(self, config_manager):
        # Call the function and check the returned value
        assert config_manager._create_empty_data_cfg() == {
            "data": {
                "train": {"ann-files": None, "data-roots": None},
                "val": {"ann-files": None, "data-roots": None},
                "test": {"ann-files": None, "data-roots": None},
                "unlabeled": {"file-list": None, "data-roots": None},
            }
        }

    @e2e_pytest_unit
    def test_export_data_cfg(self, mocker, config_manager):
        # Mock data
        data_cfg = {
            "data": {
                "train": {"ann-files": "path/to/train/ann", "data-roots": "path/to/train/images"},
                "val": {"ann-files": "path/to/val/ann", "data-roots": "path/to/val/images"},
                "test": {"ann-files": "path/to/test/ann", "data-roots": "path/to/test/images"},
                "unlabeled": {"file-list": "path/to/unlabeled/files", "data-roots": "path/to/unlabeled/images"},
            }
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            output_path = temp_file.name

        # Mock write_text function
        mock_write_text = mocker.patch("pathlib.Path.write_text")

        # Test the function
        config_manager._export_data_cfg(data_cfg, output_path)

        # Assertions
        mock_write_text.assert_called_once_with(OmegaConf.to_yaml(data_cfg), encoding="utf-8")

    @e2e_pytest_unit
    def test_build_workspace(self, mocker):
        # Setup
        task_type = "CLASSIFICATION"
        train_type = "Semisupervised"
        workspace_path = "./otx-workspace"
        args = mocker.Mock()
        args.autosplit = None
        args.workspace = workspace_path
        config_manager = ConfigManager(args)
        template = self.get_default_template(config_manager.otx_root, task_type)
        config_manager.template = template
        config_manager.train_type = train_type
        config_manager.task_type = task_type

        pathlib_mkdir_mock = mocker.patch("pathlib.Path.mkdir")
        pathlib_exists_mock = mocker.patch("pathlib.Path.exists", return_value=True)

        set_workspace_mock = mocker.patch("otx.cli.manager.config_manager.set_workspace", return_value=workspace_path)

        parse_model_template_mock = mocker.patch(
            "otx.cli.manager.config_manager.parse_model_template", return_value=template
        )

        gen_params_dict_from_args_mock = mocker.patch(
            "otx.cli.manager.config_manager.gen_params_dict_from_args", return_value={}
        )
        omageconf_load_mock = mocker.patch("otx.cli.manager.config_manager.OmegaConf.load", return_value=template)
        omageconf_merge_mock = mocker.patch("otx.cli.manager.config_manager.OmegaConf.merge", return_value=template)
        path_write_text_mock = mocker.patch("otx.cli.manager.config_manager.Path.write_text")

        config_manager_check_workspace_mock = mocker.patch(
            "otx.cli.manager.config_manager.ConfigManager.check_workspace", return_value=False
        )

        config_manager_copy_config_files_mock = mocker.patch(
            "otx.cli.manager.config_manager.ConfigManager._copy_config_files"
        )
        shutil_copyfile_mock = mocker.patch("shutil.copyfile")

        # Run
        config_manager.build_workspace()

        # Check
        set_workspace_mock.assert_called_once_with(task=task_type)
        parse_model_template_mock.assert_called_once_with(str(config_manager.workspace_root / "template.yaml"))

        # Calls
        pathlib_mkdir_mock.assert_called()
        pathlib_exists_mock.assert_called()

        gen_params_dict_from_args_mock.assert_called_once_with(args)
        omageconf_load_mock.assert_called()
        omageconf_merge_mock.assert_called()
        path_write_text_mock.assert_called()

        config_manager_check_workspace_mock.assert_called()
        config_manager_copy_config_files_mock.assert_called()

        shutil_copyfile_mock.assert_called()

    @e2e_pytest_unit
    def test_update_data_config(self, config_manager, tmp_dir_path):
        data_yaml = {
            "data": {
                "train": {"data-roots": "/path/to/train/data"},
                "val": {"data-roots": "/path/to/val/data"},
                "test": {"data-roots": "/path/to/test/data"},
                "unlabeled": {"file-list": "/path/to/unlabeled/filelist", "data-roots": "/path/to/unlabeled/data"},
            }
        }
        data_yaml_path = tmp_dir_path / "data.yaml"
        OmegaConf.save(data_yaml, str(data_yaml_path))

        config_manager.update_data_config(OmegaConf.load(str(data_yaml_path)))
        assert config_manager.data_config == {
            "train_subset": {"data_root": "/path/to/train/data"},
            "val_subset": {"data_root": "/path/to/val/data"},
            "test_subset": {"data_root": "/path/to/test/data"},
            "unlabeled_subset": {
                "data_root": "/path/to/unlabeled/data",
                "file_list": "/path/to/unlabeled/filelist",
            },
        }

        data_yaml["data"]["train"]["data-roots"] = None
        data_yaml_path = tmp_dir_path / "data.yaml"
        OmegaConf.save(data_yaml, str(data_yaml_path))

        config_manager.update_data_config(OmegaConf.load(str(data_yaml_path)))
        assert config_manager.data_config == {
            "train_subset": {"data_root": "/path/to/train/data"},
            "val_subset": {"data_root": "/path/to/val/data"},
            "test_subset": {"data_root": "/path/to/test/data"},
            "unlabeled_subset": {
                "data_root": "/path/to/unlabeled/data",
                "file_list": "/path/to/unlabeled/filelist",
            },
        }

        data_yaml["data"].pop("unlabeled")
        data_yaml_path = tmp_dir_path / "data.yaml"
        OmegaConf.save(data_yaml, str(data_yaml_path))

        config_manager.update_data_config(OmegaConf.load(str(data_yaml_path)))
        assert config_manager.data_config == {
            "train_subset": {"data_root": "/path/to/train/data"},
            "val_subset": {"data_root": "/path/to/val/data"},
            "test_subset": {"data_root": "/path/to/test/data"},
            "unlabeled_subset": {
                "data_root": "/path/to/unlabeled/data",
                "file_list": "/path/to/unlabeled/filelist",
            },
        }

    @e2e_pytest_unit
    def test_get_hyparams_config(self, mocker):
        mock_hyper_parameters = {
            "learning_rate": {
                "type": "FLOAT",
                "default_value": "0.01",
                "max_value": "1.0",
                "min_value": "0.0",
                "affects_outcome_of": ["TRAINING"],
            },
            "batch_size": {
                "type": "INTEGER",
                "default_value": "16",
                "max_value": "128",
                "min_value": "1",
                "affects_outcome_of": ["TRAINING", "TESTING"],
            },
        }
        mock_template = DictConfig({"hyper_parameters": DictConfig({"data": mock_hyper_parameters})})

        parser = argparse.ArgumentParser()
        parser.add_argument("--template")
        parser.add_argument(
            "--learning_rate",
            dest="params.learning_rate",
        )
        parser.add_argument(
            "--batch_size",
            dest="params.batch_size",
        )
        mock_input = ["--learning_rate", "0.5", "--batch_size", "8"]
        mock_args = parser.parse_args(mock_input)
        expected_hyper_parameters = {
            "learning_rate": {
                "type": "FLOAT",
                "default_value": "0.01",
                "max_value": "1.0",
                "min_value": "0.0",
                "affects_outcome_of": ["TRAINING"],
                "value": "0.5",
            },
            "batch_size": {
                "type": "INTEGER",
                "default_value": "16",
                "max_value": "128",
                "min_value": "1",
                "affects_outcome_of": ["TRAINING", "TESTING"],
                "value": "8",
            },
        }
        mock_create = mocker.patch("otx.cli.manager.config_manager.create", return_value=expected_hyper_parameters)

        config_manager = ConfigManager(mock_args)
        config_manager.template = mock_template
        config_manager.get_hyparams_config()
        mock_create.assert_called_once_with(expected_hyper_parameters)

    @e2e_pytest_unit
    def test_data_config_file_path(self, mocker, tmp_dir_path):
        parser = argparse.ArgumentParser()
        parser.add_argument("--template")
        parser.add_argument("--data")
        args = parser.parse_args([])
        config_manager = ConfigManager(args)

        # set up test workspace
        workspace_root = tmp_dir_path / "test_data_config"
        config_manager.workspace_root = workspace_root
        # workspace_root.mkdir(exist_ok=True, parents=True)
        assert config_manager.data_config_file_path == workspace_root / "data.yaml"

        # expected file path
        mock_exists = mocker.patch("otx.cli.manager.config_manager.Path.exists", return_value=False)
        expected_file_path = tmp_dir_path / "data.yaml"
        args = parser.parse_args(["--data", str(expected_file_path)])
        config_manager.args = args
        with pytest.raises(FileNotExistError):
            config_manager.data_config_file_path

        mock_exists.return_value = True
        assert config_manager.data_config_file_path == expected_file_path

    @e2e_pytest_unit
    def test_configure_template(self, mocker):
        # Given
        mock_args = mocker.MagicMock()
        mock_args.train_data_roots = ["/path/to/train/data"]
        mock_args.template = None
        mock_workspace_root = mocker.MagicMock()
        mock_workspace_root.exists.return_value = True

        mock_template = DictConfig({"name": "template_name", "task_type": "CLASSIFICATION"})
        mock_check_workspace = mocker.patch(
            "otx.cli.manager.config_manager.ConfigManager.check_workspace", return_value=True
        )
        mocker.patch("otx.cli.manager.config_manager.ConfigManager._get_template", return_value=mock_template)
        mocker.patch("otx.cli.manager.config_manager.ConfigManager._get_train_type", return_value="Incremental")
        mock_parse_model_template = mocker.patch(
            "otx.cli.manager.config_manager.parse_model_template", return_value=mock_template
        )

        config_manager = ConfigManager(args=mock_args, workspace_root=mock_workspace_root)

        # When
        config_manager.configure_template()

        # Then
        assert config_manager.task_type == "CLASSIFICATION"
        assert config_manager.model == "template_name"
        assert config_manager.train_type == "Incremental"

        config_manager.mode = "build"
        mocker.patch("otx.cli.manager.config_manager.ConfigManager._check_rebuild", return_value=True)
        config_manager.configure_template()
        assert config_manager.rebuild
        assert config_manager.task_type == "CLASSIFICATION"
        assert config_manager.model == "template_name"
        assert config_manager.train_type == "Incremental"

        mock_check_workspace.return_value = False
        mocker.patch("pathlib.Path.exists", return_value=True)
        config_manager.template = "test/template"
        config_manager.configure_template()
        mock_parse_model_template.assert_called_with("test/template")

        config_manager.template = None
        config_manager.task_type = None
        mock_check_workspace = mocker.patch(
            "otx.cli.manager.config_manager.ConfigManager.auto_task_detection", return_value="CLASSIFICATION"
        )
        config_manager.configure_template()
        assert config_manager.task_type == "CLASSIFICATION"
        assert config_manager.model == "template_name"
        assert config_manager.train_type == "Incremental"

    @e2e_pytest_unit
    def test__check_rebuild(self, mocker):
        mock_template = mocker.MagicMock()
        mock_template.task_type = "CLASSIFICATION"

        mock_args = mocker.MagicMock()
        mock_args.mode = "build"
        mock_args.task = "DETECTION"
        mock_args.template = mock_template

        config_manager = ConfigManager(mock_args)
        with pytest.raises(NotSupportedError):
            config_manager._check_rebuild()

        config_manager.template.task_type = "DETECTION"
        config_manager.args.model = None
        config_manager.args.train_type = ""
        assert not config_manager._check_rebuild()

        config_manager.args.model = "SSD"
        config_manager.template.name = "ATSS"
        config_manager.args.train_type = "Semisupervised"
        assert config_manager._check_rebuild()

    @e2e_pytest_unit
    def test_configure_data_config(self, mocker):
        data_yaml = {
            "data": {
                "train": {"ann-files": None, "data-roots": "train/data/roots"},
                "val": {"ann-files": None, "data-roots": None},
                "test": {"ann-files": None, "data-roots": None},
                "unlabeled": {"file-list": None, "data-roots": None},
            }
        }
        mock_configure_dataset = mocker.patch(
            "otx.cli.manager.config_manager.configure_dataset", return_value=data_yaml
        )
        mock_auto_split = mocker.patch("otx.cli.manager.config_manager.ConfigManager.auto_split_data", return_value={})
        mock_get_data_yaml = mocker.patch(
            "otx.cli.manager.config_manager.ConfigManager._get_arg_data_yaml", return_value=data_yaml
        )
        mock_save_data = mocker.patch("otx.cli.manager.config_manager.ConfigManager._save_data")
        mock_export_data_cfg = mocker.patch("otx.cli.manager.config_manager.ConfigManager._export_data_cfg")
        mock_update_data_config = mocker.patch("otx.cli.manager.config_manager.ConfigManager.update_data_config")

        mock_args = mocker.MagicMock()
        mock_args.mode = "build"

        config_manager = ConfigManager(mock_args)
        config_manager.train_type = "Incremental"
        config_manager.configure_data_config(update_data_yaml=True)

        mock_configure_dataset.assert_called_once()
        mock_auto_split.assert_called_once_with("train/data/roots", "")
        mock_get_data_yaml.assert_called_once()
        mock_save_data.assert_called_once()
        mock_export_data_cfg.assert_called_once()
        mock_update_data_config.assert_called_once_with(data_yaml)

    @e2e_pytest_unit
    def test__get_train_type(self, mocker):
        mock_args = mocker.MagicMock()
        mock_params_dict = {"algo_backend": {"train_type": {"value": "Semisupervised"}}}
        mock_configure_dataset = mocker.patch(
            "otx.cli.manager.config_manager.gen_params_dict_from_args", return_value=mock_params_dict
        )
        config_manager = ConfigManager(args=mock_args)
        config_manager.mode = "build"
        assert config_manager._get_train_type() == "Semisupervised"

        config_manager.args.train_type = "Incremental"
        mock_configure_dataset.return_value = {}
        assert config_manager._get_train_type() == "Incremental"

        mock_template = mocker.MagicMock()
        mock_template.hyper_parameters.parameter_overrides = {
            "algo_backend": {"train_type": {"default_value": "Selfsupervised"}}
        }
        config_manager.template = mock_template
        assert config_manager._get_train_type(ignore_args=True) == "Selfsupervised"

        config_manager.template.hyper_parameters.parameter_overrides = {}
        assert config_manager._get_train_type(ignore_args=True) == "Incremental"

    @e2e_pytest_unit
    def test_auto_task_detection(self, mocker):
        mock_args = mocker.MagicMock()
        config_manager = ConfigManager(args=mock_args)
        with pytest.raises(CliException):
            config_manager.auto_task_detection("")

        mock_get_data_format = mocker.patch(
            "otx.cli.manager.config_manager.DatasetManager.get_data_format", return_value="Unexpected"
        )
        with pytest.raises(ConfigValueError):
            config_manager.auto_task_detection("data/roots")

        mock_get_data_format.return_value = "coco"
        assert config_manager.auto_task_detection("data/roots") == "DETECTION"

    @e2e_pytest_unit
    def test_auto_split_data(self, mocker):
        mock_get_data_format = mocker.patch(
            "otx.cli.manager.config_manager.DatasetManager.get_data_format", return_value="coco"
        )
        mock_import_dataset = mocker.patch(
            "otx.cli.manager.config_manager.DatasetManager.import_dataset", return_value=None
        )
        mock_get_train_dataset = mocker.patch(
            "otx.cli.manager.config_manager.DatasetManager.get_train_dataset", return_value="train_dataset"
        )
        mock_get_val_dataset = mocker.patch(
            "otx.cli.manager.config_manager.DatasetManager.get_val_dataset", return_value="val_dataset"
        )
        mock_auto_split = mocker.patch(
            "otx.cli.manager.config_manager.DatasetManager.auto_split",
            return_value={"train": "auto_train", "val": "auto_val"},
        )

        mock_args = mocker.MagicMock()
        config_manager = ConfigManager(args=mock_args)
        assert config_manager.auto_split_data("test_data_root", task="DETECTION") == {
            "train": "train_dataset",
            "val": "val_dataset",
        }

        mock_get_val_dataset.return_value = None
        assert config_manager.auto_split_data("test_data_root", task="DETECTION") == {
            "train": "auto_train",
            "val": "auto_val",
        }

        mock_get_data_format.return_value = "Unexpected"
        assert config_manager.auto_split_data("test_data_root", task="DETECTION") is None
        mock_get_data_format.assert_called()
        mock_import_dataset.assert_called()
        mock_get_train_dataset.assert_called()
        mock_get_val_dataset.assert_called()
        mock_auto_split.assert_called()

    @e2e_pytest_unit
    def test_get_dataset_config(self, mocker):
        mock_args = mocker.MagicMock()
        config_manager = ConfigManager(args=mock_args)
        config_manager.task_type = "DETECTION"
        config_manager.data_config = {
            "train_subset": {"data_root": "train_path"},
            "val_subset": {"data_root": "val_path"},
            "test_subset": {"data_root": "test_path"},
        }
        dataset_config = config_manager.get_dataset_config(["train", "val", "test"])
        assert dataset_config["task_type"] == "DETECTION"
        assert "train_data_roots" in dataset_config
        assert "val_data_roots" in dataset_config
        assert "test_data_roots" in dataset_config
