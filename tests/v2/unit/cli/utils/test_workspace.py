# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict

import pytest
from omegaconf import OmegaConf
from otx.v2.cli.utils.workspace import Workspace, set_workspace
from pytest_mock import MockerFixture


@pytest.fixture()
def workspace(tmp_dir_path: Path) -> Workspace:
    return Workspace(work_dir=tmp_dir_path)


def test_set_workspace(tmp_dir_path: Path) -> None:
    assert set_workspace() == "./otx-workspace"
    assert set_workspace(root=str(tmp_dir_path)) == str(tmp_dir_path / "otx-workspace")

class TestWorkspace:
    def test_workspace_init(self, workspace: Workspace, tmp_dir_path: Path) -> None:
        assert workspace.otx_root is not None
        assert workspace.work_dir.exists()

        workspace = Workspace(work_dir=tmp_dir_path, task="classification")
        assert workspace.work_dir == tmp_dir_path/"classification"
        assert workspace.work_dir.exists()


    def test_workspace_dump_config(self, workspace: Workspace, mocker: MockerFixture) -> None:
        config: Dict[str, str] = {"foo": "bar"}
        workspace.dump_config(config=config)
        assert workspace.config_path.exists()
        assert OmegaConf.load(workspace.config_path) == config
        assert workspace.check_workspace()

        workspace.dump_config(config=config, filename=str(workspace.work_dir / "test.yaml"))
        assert OmegaConf.load(workspace.work_dir / "test.yaml") == config

        workspace._config = workspace.work_dir / "test.yaml"
        workspace.dump_config()
        mocker.patch("otx.v2.cli.utils.workspace.Path.is_file", return_value=False)
        with pytest.raises(FileNotFoundError):
            workspace.dump_config()

    def test_workspace_add_config(self, workspace: Workspace) -> None:
        config: Dict[str, str] = {"foo": "bar"}
        workspace.add_config(config)
        assert workspace._config == config


    def test_workspace_update_latest(self, workspace: Workspace, mocker: MockerFixture) -> None:
        mock_symlink_to = mocker.patch("otx.v2.cli.utils.workspace.Path.symlink_to")
        workspace.latest_dir = workspace.work_dir / "sub_folder" / "latest"
        target_dir = workspace.work_dir / "models"
        target_dir.mkdir(exist_ok=True)
        workspace.update_latest(target_dir)
        mock_symlink_to.assert_called_once_with(target_dir.resolve())

        mock_exists = mocker.patch("otx.v2.cli.utils.workspace.Path.exists", return_value=True)
        mock_unlink = mocker.patch("otx.v2.cli.utils.workspace.Path.unlink")
        workspace.update_latest(target_dir)
        mock_exists.assert_called_once()
        mock_unlink.assert_called_once()
