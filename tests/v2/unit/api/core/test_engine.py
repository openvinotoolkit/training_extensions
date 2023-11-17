"""OTX V2 API-core Unit-Test codes (Engine)."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from otx.v2.api.core.engine import Engine
from otx.v2.api.entities.task_type import TaskType
from pytest_mock.plugin import MockerFixture


class TestEngine:
    def test_init(self, tmp_dir_path: Path) -> None:
        """Test the initialization of the Engine class.

        Steps:
        1. Create an instance of the Engine class with the given temporary directory path.
        2. Verify that the work_dir attribute of the engine instance is set to the given temporary directory path.
        3. Verify that the registry name attribute of the engine instance is set to "base".
        4. Verify that the timestamp attribute of the engine instance is not None.

        Args:
        ----
            tmp_dir_path (Path): A temporary directory path.

        Returns:
        -------
            None
        """
        engine = Engine(work_dir=tmp_dir_path, task=TaskType.CLASSIFICATION)

        assert engine.work_dir == tmp_dir_path
        assert engine.registry.name == "base"
        assert engine.timestamp is not None

    def test_train(self, mocker: MockerFixture) -> None:
        """Test the train method of the Engine class.

        Steps:
        1. Create an instance of the Engine class with a None work_dir attribute.
        2. Mock the train method of the Engine class.
        3. Call the train method of the engine instance with None arguments.
        4. Verify that the mock train method was called once.

        Args:
        ----
            mocker (MockerFixture): A pytest fixture for mocking.

        Returns:
        -------
            None
        """
        engine = Engine(work_dir=None, task=TaskType.CLASSIFICATION)
        mock_train = mocker.spy(Engine, "train")
        engine.train(None, None)

        mock_train.assert_called()
        assert mock_train.call_count == 1

    def test_validate(self, mocker: MockerFixture) -> None:
        """Test the validate method of the Engine class.

        Steps:
        1. Create an instance of the Engine class with a None work_dir attribute.
        2. Mock the validate method of the Engine class.
        3. Call the validate method of the engine instance with None arguments.
        4. Verify that the mock validate method was called once.

        Args:
        ----
            mocker (MockerFixture): A pytest fixture for mocking.

        Returns:
        -------
            None
        """
        engine = Engine(work_dir=None, task=TaskType.CLASSIFICATION)
        mock_validate = mocker.spy(Engine, "validate")
        engine.validate(None, None)

        mock_validate.assert_called()
        assert mock_validate.call_count == 1

    def test_test(self, mocker: MockerFixture) -> None:
        """Test the test method of the Engine class.

        Steps:
        1. Create an instance of the Engine class with a None work_dir attribute.
        2. Mock the test method of the Engine class.
        3. Call the test method of the engine instance with None arguments.
        4. Verify that the mock test method was called once.

        Args:
        ----
            mocker (MockerFixture): A pytest fixture for mocking.

        Returns:
        -------
            None
        """
        engine = Engine(work_dir=None, task=TaskType.CLASSIFICATION)
        mock_test = mocker.spy(Engine, "test")
        engine.test(None, None)

        mock_test.assert_called()
        assert mock_test.call_count == 1

    def test_predict(self, mocker: MockerFixture) -> None:
        """Test the predict method of the Engine class.

        Steps:
        1. Create an instance of the Engine class with a None work_dir attribute.
        2. Mock the predict method of the Engine class.
        3. Call the predict method of the engine instance with None arguments.
        4. Verify that the mock predict method was called once.

        Args:
        ----
            mocker (MockerFixture): A pytest fixture for mocking.

        Returns:
        -------
            None
        """
        engine = Engine(work_dir=None, task=TaskType.CLASSIFICATION)
        mock_predict = mocker.spy(Engine, "predict")
        engine.predict(None, None)

        mock_predict.assert_called()
        assert mock_predict.call_count == 1

    def test_export(self, mocker: MockerFixture) -> None:
        """Test the export method of the Engine class.

        Steps:
        1. Create an instance of the Engine class with a None work_dir attribute.
        2. Mock the export method of the Engine class.
        3. Call the export method of the engine instance.
        4. Verify that the mock export method was called once.

        Args:
        ----
            mocker (MockerFixture): A pytest fixture for mocking.

        Returns:
        -------
            None
        """
        engine = Engine(work_dir=None, task=TaskType.CLASSIFICATION)
        mock_export = mocker.spy(Engine, "export")
        engine.export()

        mock_export.assert_called()
        assert mock_export.call_count == 1
