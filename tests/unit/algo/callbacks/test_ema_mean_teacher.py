# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from unittest.mock import MagicMock, patch

import pytest
from otx.algo.callbacks.ema_mean_teacher import EMAMeanTeacher


class TestEMAMeanTeacher:
    @pytest.fixture()
    def ema_mean_teacher(self):
        return EMAMeanTeacher(momentum=0.99, start_epoch=1)

    def test_initialization(self, ema_mean_teacher):
        assert ema_mean_teacher.momentum == 0.99
        assert ema_mean_teacher.start_epoch == 1
        assert not ema_mean_teacher.synced_models

    @patch("otx.algo.callbacks.ema_mean_teacher.Trainer")
    @patch("otx.algo.callbacks.ema_mean_teacher.LightningModule")
    def test_on_train_start(self, mock_trainer, mock_pl_module, ema_mean_teacher):
        mock_model = MagicMock()
        mock_model.student_model = MagicMock()
        mock_model.teacher_model = MagicMock()
        mock_trainer.model.model = mock_model

        ema_mean_teacher.on_train_start(mock_trainer, mock_pl_module)

        assert ema_mean_teacher.src_model is not None
        assert ema_mean_teacher.dst_model is not None
        assert ema_mean_teacher.src_model == mock_model.student_model
        assert ema_mean_teacher.dst_model == mock_model.teacher_model

    @patch("otx.algo.callbacks.ema_mean_teacher.Trainer")
    @patch("otx.algo.callbacks.ema_mean_teacher.LightningModule")
    def test_on_train_batch_end(self, mock_trainer, mock_pl_module, ema_mean_teacher):
        mock_trainer.current_epoch = 2
        mock_trainer.global_step = 10

        ema_mean_teacher.src_params = {"param": MagicMock(requires_grad=True)}
        ema_mean_teacher.dst_params = {"param": MagicMock(requires_grad=True)}

        ema_mean_teacher.on_train_batch_end(mock_trainer, mock_pl_module, None, None, None)
        assert ema_mean_teacher.synced_models is True
        assert ema_mean_teacher.dst_params["param"].data.copy_.call_count == 2  # 1 for copy and 1 for ema

    def test_copy_model(self, ema_mean_teacher):
        src_param = MagicMock(requires_grad=True)
        dst_param = MagicMock(requires_grad=True)
        ema_mean_teacher.src_params = {"param": src_param}
        ema_mean_teacher.dst_params = {"param": dst_param}

        ema_mean_teacher._copy_model()

        dst_param.data.copy_.assert_called_once_with(src_param.data)

    def test_ema_model(self, ema_mean_teacher):
        src_param = MagicMock(requires_grad=True)
        dst_param = MagicMock(requires_grad=True)
        ema_mean_teacher.src_params = {"param": src_param}
        ema_mean_teacher.dst_params = {"param": dst_param}
        ema_mean_teacher.synced_models = True
        ema_mean_teacher._ema_model(global_step=10)

        momentum = min(1 - 1 / (10 + 1), ema_mean_teacher.momentum)
        expected_value = dst_param.data * momentum + src_param.data * (1 - momentum)
        dst_param.data.copy_.assert_called_once_with(expected_value)
