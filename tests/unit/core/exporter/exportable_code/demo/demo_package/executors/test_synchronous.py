# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of AsyncExecutor in demo_package."""

from unittest.mock import MagicMock

import pytest

target_file = None
SyncExecutor = None


@pytest.fixture(scope="module", autouse=True)
def fxt_import_module():
    global target_file  # noqa: PLW0603
    global SyncExecutor  # noqa: PLW0603
    from otx.core.exporter.exportable_code.demo.demo_package.executors import synchronous
    from otx.core.exporter.exportable_code.demo.demo_package.executors.synchronous import SyncExecutor as Cls1

    target_file = synchronous
    SyncExecutor = Cls1


class TestSyncExecutor:
    @pytest.fixture()
    def mock_model(self):
        return MagicMock(side_effect=lambda x: (x, x))

    @pytest.fixture()
    def mock_visualizer(self):
        visualizer = MagicMock()
        visualizer.is_quit.return_value = False
        visualizer.draw.side_effect = lambda x, _: x
        return visualizer

    def test_init(self, mock_model, mock_visualizer):
        SyncExecutor(mock_model, mock_visualizer)

    @pytest.fixture()
    def mock_streamer(self, mocker):
        return mocker.patch.object(target_file, "get_streamer", return_value=range(3))

    @pytest.fixture()
    def mock_dump_frames(self, mocker):
        return mocker.patch.object(target_file, "dump_frames")

    def test_run(self, mock_model, mock_visualizer, mock_streamer, mock_dump_frames):
        executor = SyncExecutor(mock_model, mock_visualizer)
        mock_input_stream = MagicMock()
        executor.run(mock_input_stream, MagicMock())

        mock_model.assert_called()
        for i in range(3):
            assert mock_model.call_args_list[i] == ((i,),)
        mock_visualizer.draw.assert_called()
        for i in range(3):
            assert mock_visualizer.draw.call_args_list[i] == ((i, i),)
        mock_visualizer.show.assert_called()
        for i in range(3):
            assert mock_visualizer.show.call_args_list[i] == ((i,),)
        mock_dump_frames.assert_called_once_with(
            list(range(3)),
            mock_visualizer.output,
            mock_input_stream,
            range(3),
        )
