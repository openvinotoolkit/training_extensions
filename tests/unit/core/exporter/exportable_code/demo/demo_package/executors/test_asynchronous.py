# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of AsyncExecutor in demo_package."""

from unittest.mock import MagicMock

import pytest

target_file = None
AsyncExecutor = None


@pytest.fixture(scope="module", autouse=True)
def fxt_import_module():
    global target_file  # noqa: PLW0603
    global AsyncExecutor  # noqa: PLW0603
    from otx.core.exporter.exportable_code.demo.demo_package.executors import asynchronous
    from otx.core.exporter.exportable_code.demo.demo_package.executors.asynchronous import AsyncExecutor as Cls1

    target_file = asynchronous
    AsyncExecutor = Cls1


class MockAsyncPipeline:
    def __init__(self, *args, **kwargs):
        self.arr = []
        self.idx = 0

    def get_result(self, *args, **kwawrgs):
        if self.idx >= len(self.arr):
            return None
        ret = self.arr[self.idx]
        self.idx += 1
        return ret

    def submit_data(self, frame, *args, **kwawrgs):
        self.arr.append(frame)

    def await_all(self):
        pass


class TestAsyncExecutor:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        mocker.patch.object(target_file, "AsyncPipeline", side_effect=MockAsyncPipeline)

    @pytest.fixture()
    def mock_model(self):
        return MagicMock()

    @pytest.fixture()
    def mock_visualizer(self):
        visualizer = MagicMock()
        visualizer.is_quit.return_value = False
        visualizer.draw.side_effect = lambda x, _: x
        return visualizer

    def test_init(self, mock_model, mock_visualizer):
        AsyncExecutor(mock_model, mock_visualizer)

    @pytest.fixture()
    def mock_streamer(self, mocker):
        return mocker.patch.object(target_file, "get_streamer", return_value=range(1, 4))

    @pytest.fixture()
    def mock_dump_frames(self, mocker):
        return mocker.patch.object(target_file, "dump_frames")

    def test_run(self, mocker, mock_model, mock_visualizer, mock_streamer, mock_dump_frames):
        mock_render_result = mocker.patch.object(AsyncExecutor, "render_result", side_effect=lambda x: x)
        executor = AsyncExecutor(mock_model, mock_visualizer)
        executor.run(MagicMock())

        mock_render_result.assert_called()
        for i in range(1, 4):
            assert mock_render_result.call_args_list[i - 1].args == (i,)
        mock_visualizer.show.assert_called()
        for i in range(1, 4):
            assert mock_visualizer.show.call_args_list[i - 1].args == (i,)
        mock_dump_frames.assert_called()

    def test_render_result(self, mock_model, mock_visualizer):
        executor = AsyncExecutor(mock_model, mock_visualizer)
        mock_pred = MagicMock()
        cur_frame = MagicMock()
        frame_meta = {"frame": cur_frame}
        fake_results = (mock_pred, frame_meta)
        executor.render_result(fake_results)

        mock_visualizer.draw.assert_called_once_with(cur_frame, mock_pred)
