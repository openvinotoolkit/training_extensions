"""OTX V2 API-utils Unit-Test codes (auto_utils)."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path
from types import ModuleType

import pytest
from _pytest.monkeypatch import MonkeyPatch
from otx.v2.api.utils.auto_utils import (
    check_semisl_requirements,
    configure_task_type,
    configure_train_type,
    count_imgs_in_dir,
)
from pytest_mock.plugin import MockerFixture


class TestAutoUtils:
    def test_configure_task_type(self, mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
        # Test for valid data format
        with mocker.patch(
            "otx.v2.adapters.datumaro.manager.dataset_manager.DatasetManager.get_data_format", return_value="imagenet",
        ):
            task, data_format = configure_task_type("data/roots")
            assert task == "CLASSIFICATION"
            assert data_format == "imagenet"

        with mocker.patch(
            "otx.v2.adapters.datumaro.manager.dataset_manager.DatasetManager.get_data_format", return_value="no_format",
        ), pytest.raises(ValueError, match="Can't find proper task"):
            configure_task_type("data/roots")

        def mock_import(name: str, *args) -> ModuleType:
            if name == "otx.v2.adapters.datumaro.manager.dataset_manager":
                msg = f"No module named {name}"
                raise ImportError(msg)
            return importlib.import_module(name, *args)

        monkeypatch.setattr("builtins.__import__", mock_import)
        with pytest.raises(ImportError, match="Need datumaro to automatically detect the task type."):
            configure_task_type(data_roots="data/roots", data_format=None)

    def test_count_imgs_in_dir(self, tmp_dir_path: Path) -> None:
        # Create a temporary directory with some image files
        tmp_dir = tmp_dir_path / "test_count_imgs_in_dir"
        tmp_dir.mkdir()
        (tmp_dir / "image1.jpg").touch()
        (tmp_dir / "image2.png").touch()
        (tmp_dir / "subdir").mkdir()
        (tmp_dir / "subdir" / "image3.jpeg").touch()

        # Test with non-recursive search
        assert count_imgs_in_dir(tmp_dir) == 2

        # Test with recursive search
        assert count_imgs_in_dir(tmp_dir, recursive=True) == 3


    def test_check_semisl_requirements(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        # Test with valid unlabeled directory
        assert not check_semisl_requirements(None)

        with mocker.patch("otx.v2.api.utils.auto_utils.count_imgs_in_dir", return_value=0):
            assert not check_semisl_requirements(tmp_dir_path)

        # Test with invalid unlabeled directory
        with pytest.raises(ValueError, match="unlabeled-data-roots isn't a directory"):
            check_semisl_requirements("path/no_data")

        # Test with too few images in unlabeled directory
        with mocker.patch("otx.v2.api.utils.auto_utils.count_imgs_in_dir", return_value=2):
            assert check_semisl_requirements(tmp_dir_path) == tmp_dir_path


    def test_configure_train_type(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        with mocker.patch("otx.v2.api.utils.auto_utils.count_imgs_in_dir", return_value=0):
            assert configure_train_type(train_data_roots=tmp_dir_path, unlabeled_data_roots=None) == "Incremental"

        with mocker.patch("otx.v2.api.utils.auto_utils.count_imgs_in_dir", return_value=1):
            assert configure_train_type(train_data_roots=tmp_dir_path, unlabeled_data_roots=None) == "Selfsupervised"

        with mocker.patch("otx.v2.api.utils.auto_utils.count_imgs_in_dir", return_value=0), mocker.patch("otx.v2.api.utils.auto_utils.check_semisl_requirements", return_value="path/to/unlabeled"):
            assert configure_train_type(train_data_roots=tmp_dir_path, unlabeled_data_roots=None) == "Semisupervised"

        assert configure_train_type(train_data_roots=None, unlabeled_data_roots=None) is None
