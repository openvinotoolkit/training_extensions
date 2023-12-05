# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from otx.algo.callbacks.iteration_timer import IterationTimer


class TestIterationTimer:
    @pytest.mark.parametrize("phase", ["train", "validation", "test"])
    @patch("otx.algo.callbacks.iteration_timer.time")
    def test_all_phases(self, mock_time, phase) -> None:
        mock_trainer = MagicMock()
        mock_pl_module = MagicMock()
        mock_batch = MagicMock()
        batch_size = 64
        mock_batch.batch_size = batch_size
        mock_outputs = MagicMock()

        timer = IterationTimer()

        batch_len = 2

        # Timestamp
        # 0, batch_start
        # 1, batch_end
        # 2, ...
        # 3,
        mock_time.side_effect = range(2 * batch_len)

        for batch_idx in range(batch_len):
            getattr(timer, f"on_{phase}_batch_start")(
                trainer=mock_trainer,
                pl_module=mock_pl_module,
                batch=mock_batch,
                batch_idx=batch_idx,
            )

            if batch_idx == 0:
                assert (
                    not mock_pl_module.log.called
                ), "Cannot log data and iter time at the first batch step"
            else:
                mock_pl_module.log.assert_called_with(
                    name=f"{phase}/data_time",
                    value=1,
                    prog_bar=timer.prog_bar,
                    on_step=timer.on_step,
                    on_epoch=timer.on_epoch,
                    batch_size=batch_size,
                )

            getattr(timer, f"on_{phase}_batch_end")(
                trainer=mock_trainer,
                pl_module=mock_pl_module,
                outputs=mock_outputs,
                batch=mock_batch,
                batch_idx=batch_idx,
            )

            assert timer.start_time[phase] == 2 * batch_idx
            assert timer.end_time[phase] == 2 * batch_idx + 1

            if batch_idx == 0:
                assert (
                    not mock_pl_module.log.called
                ), "Cannot log data and iter time at the first batch step"
            else:
                mock_pl_module.log.assert_called_with(
                    name=f"{phase}/iter_time",
                    value=2,
                    prog_bar=timer.prog_bar,
                    on_step=timer.on_step,
                    on_epoch=timer.on_epoch,
                    batch_size=batch_size,
                )
