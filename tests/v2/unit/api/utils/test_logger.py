"""OTX V2 API-utils Unit-Test codes (Loggers)."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch
from otx.v2.api.utils import logger
from pytest_mock.plugin import MockerFixture


class TestLogger:
    @pytest.fixture()
    def log_file(self, tmp_dir_path: Path) -> Path:
        """
        Fixture that creates a temporary log file for testing purposes.

        Args:
            tmp_dir_path (Path): Path to the temporary directory.

        Returns:
            Path: Path to the temporary log file.
        """
        path = tmp_dir_path / "test_log.log"

        def remove_file() -> None:
            Path(path).unlink()

        yield path

        remove_file()

    def test_config_logger(self, log_file: Path) -> None:
        """
        Test function that checks if the logger is properly configured.

        Args:
            log_file (Path): Path to the temporary log file.
        """
        logger.config_logger(log_file, level="DEBUG")
        test_logger = logger.get_logger()
        assert isinstance(test_logger, logging.Logger)
        test_logger.print("test_config_logger")

        with pytest.raises(TypeError, match="Level not an integer or a valid string"):
            logger.config_logger(log_file, level=None)

        with pytest.raises(ValueError, match="Log level must be one of"):
            logger.config_logger(log_file, level="11")

    def test_get_log_dir(self, log_file: Path) -> None:
        """
        Test function that checks if the log directory is properly retrieved.

        Args:
            log_file (Path): Path to the temporary log file.
        """
        logger.config_logger(log_file, level="DEBUG")
        assert logger.get_log_dir() == str(Path(log_file).parent)

    def test_local_master_only(self) -> None:
        """
        Test function that checks if the local_master_only decorator works properly.
        """
        @logger.local_master_only
        def test_func() -> str:
            return "test"

        assert test_func() == "test"

    def test_local_master_only_with_distributed(self, monkeypatch: MonkeyPatch, mocker: MockerFixture) -> None:
        """
        Test function that checks if the local_master_only decorator works properly with distributed training.

        Args:
            monkeypatch (MonkeyPatch): Pytest monkeypatch fixture.
            mocker (MockerFixture): Pytest mocker fixture.
        """
        monkeypatch.setenv("LOCAL_RANK", "1")
        mocker.patch("torch.distributed.is_available", return_value=True)
        mocker.patch("torch.distributed.is_initialized", return_value=True)
        @logger.local_master_only
        def test_func() -> str:
            return "test"

        assert test_func() is None
