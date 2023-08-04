"""OTX V2 API-utils Unit-Test codes (importing utils)."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path
from typing import Dict

import pytest
from otx.v2.api.utils.importing import (
    get_all_args,
    get_default_args,
    get_files_dict,
    get_impl_class,
    get_otx_root_path,
)
from pytest_mock import MockerFixture


class TestImportingUtils:
    def test_get_impl_class(self, mocker: MockerFixture) -> None:
        """Test the get_impl_class function in the otx.v2.api.utils.importing module."""
        # Test that get_impl_class returns the correct class
        class_path = "otx.v2.api.core.auto_runner.AutoRunner"
        task_impl_class = get_impl_class(class_path)
        assert task_impl_class.__name__ == "AutoRunner"

        # Test that get_impl_class raises an exception when the class path is invalid
        with pytest.raises(ModuleNotFoundError):
            get_impl_class("invalid.path.to.class")

        class TestModule:
            DEBUG = ValueError("Test exception")

        mocker.patch("otx.v2.api.utils.importing.importlib.import_module", return_value=TestModule)
        with pytest.raises(ValueError, match="Test exception"):
            get_impl_class("test.module.invalid_function")

    def test_get_non_default_args(self) -> None:
        """Test the get_non_default_args function in the otx.v2.api.utils.importing module."""
        # Test that get_non_default_args returns the correct non-default arguments
        def test_func(a: int, b: int, c: int=1, d: int=2) -> tuple:
            return a, b, c, d

        expected_args = [("c", 1), ("d", 2)]
        result = get_default_args(test_func)
        assert result == expected_args

        # Test that get_non_default_args returns an empty list when all arguments are non-default
        def test_func2(a: int, b: int) -> tuple:
            return a, b

        expected_args = []
        assert get_default_args(test_func2) == expected_args

    def test_get_all_args(self) -> None:
        """Test the get_all_args function in the otx.v2.api.utils.importing module."""
        # Test that get_all_args returns the correct arguments
        def test_func(a: int, b: int, c: int=1, d:int=2) -> tuple:
            return a, b, c, d

        expected_args = ["a", "b", "c", "d"]
        assert get_all_args(test_func) == expected_args

    def test_get_otx_root_path(self, mocker: MockerFixture) -> None:
        """Test the get_otx_root_path function in the otx.v2.api.utils.importing module."""
        # Test that get_otx_root_path returns the correct path
        otx = importlib.import_module("otx")
        expected_path = str(Path(otx.__file__).parent)
        assert get_otx_root_path() == expected_path

        mocker.patch("otx.v2.api.utils.importing.importlib.import_module", return_value=None)
        with pytest.raises(ModuleNotFoundError):
            get_otx_root_path()

    def test_get_files_dict(self, mocker: MockerFixture) -> None:
        """Test the get_files_dict function in the otx.v2.api.utils.importing module."""
        # Test that get_files_dict returns the correct dictionary
        folder_path = "/path/to/folder"
        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch(
            "pathlib.Path.iterdir",
            return_value=[
                Path("/path/to/folder/file1.txt"),
                Path("/path/to/folder/file2.txt"),
            ],
        )
        mocker.patch("pathlib.Path.is_file", return_value=True)

        expected_dict: Dict[str, str] = {
            "file1": "/path/to/folder/file1.txt",
            "file2": "/path/to/folder/file2.txt",
        }
        assert get_files_dict(folder_path) == expected_dict

        # Test that get_files_dict raises an exception when the folder path does not exist
        mocker.patch("pathlib.Path.exists", return_value=False)
        with pytest.raises(ValueError, match="The specified folder path does not exist."):
            get_files_dict(folder_path)
