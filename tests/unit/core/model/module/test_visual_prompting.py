# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for visual prompting module."""

from unittest.mock import Mock

import pytest
import torch
from otx.core.data.entity.visual_prompting import (
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
)
from otx.core.model.module.visual_prompting import OTXVisualPromptingLitModule, OTXZeroShotVisualPromptingLitModule
from torch import nn
from torchmetrics.aggregation import MeanMetric
from torchmetrics.collections import MetricCollection


class TestOTXVisualPromptingLitModule:
    @pytest.fixture()
    def otx_visual_prompting_lit_module(self) -> OTXVisualPromptingLitModule:
        return OTXVisualPromptingLitModule(otx_model=Mock(spec=nn.Module), torch_compile=False)

    def test_configure_metric(self, otx_visual_prompting_lit_module) -> None:
        """Test configure_metric."""
        otx_visual_prompting_lit_module.configure_metric()

        assert isinstance(otx_visual_prompting_lit_module.val_metric, MetricCollection)
        assert isinstance(otx_visual_prompting_lit_module.test_metric, MetricCollection)

    def test_log_metrics(self, mocker, otx_visual_prompting_lit_module) -> None:
        """Test _log_metrics."""
        mocker_log = mocker.patch.object(otx_visual_prompting_lit_module, "log")
        meter = MetricCollection({"mean": MeanMetric()})
        meter["mean"].update(torch.tensor(1))

        otx_visual_prompting_lit_module._log_metrics(meter, "test")

        mocker_log.assert_called()

    def test_training_step(self, mocker, otx_visual_prompting_lit_module) -> None:
        """Test training_step."""
        otx_visual_prompting_lit_module.model.return_value = {"loss": torch.tensor(1.0)}
        mocker_log_metrics = mocker.patch.object(otx_visual_prompting_lit_module, "_log_metrics")

        _ = otx_visual_prompting_lit_module.training_step({"loss": torch.tensor(1.0)}, 0)

        mocker_log_metrics.assert_called_once()

    def test_inference_step(self, mocker, otx_visual_prompting_lit_module, fxt_vpm_data_entity) -> None:
        """Test _inference_step."""
        otx_visual_prompting_lit_module.configure_metric()
        otx_visual_prompting_lit_module.model.return_value = fxt_vpm_data_entity[2]
        mocker_updates = {}
        for k, v in otx_visual_prompting_lit_module.test_metric.items():
            mocker_updates[k] = mocker.patch.object(v, "update")

        otx_visual_prompting_lit_module._inference_step(
            otx_visual_prompting_lit_module.test_metric,
            fxt_vpm_data_entity[1],
            0,
        )

        for v in mocker_updates.values():
            v.assert_called_once()


class TestOTXZeroShotVisualPromptingLitModule:
    @pytest.fixture()
    def otx_zero_shot_visual_prompting_lit_module(self) -> OTXZeroShotVisualPromptingLitModule:
        return OTXZeroShotVisualPromptingLitModule(otx_model=Mock(spec=nn.Module), torch_compile=False)

    def test_configure_metric(self, otx_zero_shot_visual_prompting_lit_module) -> None:
        """Test configure_metric."""
        otx_zero_shot_visual_prompting_lit_module.configure_metric()

        assert not hasattr(otx_zero_shot_visual_prompting_lit_module, "val_metric")
        assert isinstance(otx_zero_shot_visual_prompting_lit_module.test_metric, MetricCollection)

    def test_on_test_start(self, mocker, otx_zero_shot_visual_prompting_lit_module) -> None:
        """Test on_test_start."""
        otx_zero_shot_visual_prompting_lit_module.model.load_latest_reference_info = Mock(return_value=False)
        otx_zero_shot_visual_prompting_lit_module.trainer = Mock()
        mocker_run = mocker.patch.object(otx_zero_shot_visual_prompting_lit_module.trainer.fit_loop, "run")
        mocker_setup_data = mocker.patch.object(
            otx_zero_shot_visual_prompting_lit_module.trainer._evaluation_loop,
            "setup_data",
        )
        mocker_reset = mocker.patch.object(otx_zero_shot_visual_prompting_lit_module.trainer._evaluation_loop, "reset")

        otx_zero_shot_visual_prompting_lit_module.on_test_start()

        mocker_run.assert_called_once()
        mocker_setup_data.assert_called_once()
        mocker_reset.assert_called_once()

    def test_on_predict_start(self, mocker, otx_zero_shot_visual_prompting_lit_module) -> None:
        """Test on_predict_start."""
        otx_zero_shot_visual_prompting_lit_module.model.load_latest_reference_info = Mock(return_value=False)
        otx_zero_shot_visual_prompting_lit_module.trainer = Mock()
        mocker_run = mocker.patch.object(otx_zero_shot_visual_prompting_lit_module.trainer.fit_loop, "run")
        mocker_setup_data = mocker.patch.object(
            otx_zero_shot_visual_prompting_lit_module.trainer._evaluation_loop,
            "setup_data",
        )
        mocker_reset = mocker.patch.object(otx_zero_shot_visual_prompting_lit_module.trainer._evaluation_loop, "reset")

        otx_zero_shot_visual_prompting_lit_module.on_predict_start()

        mocker_run.assert_called_once()
        mocker_setup_data.assert_called_once()
        mocker_reset.assert_called_once()

    def test_on_train_epoch_end(self, mocker, otx_zero_shot_visual_prompting_lit_module) -> None:
        """Test on_train_epoch_end."""
        otx_zero_shot_visual_prompting_lit_module.model.save_outputs = True
        otx_zero_shot_visual_prompting_lit_module.model.save_reference_info = Mock()
        otx_zero_shot_visual_prompting_lit_module.trainer = Mock()
        mocker.patch.object(otx_zero_shot_visual_prompting_lit_module.trainer, "default_root_dir")

        otx_zero_shot_visual_prompting_lit_module.on_train_epoch_end()

    def test_inference_step(
        self,
        mocker,
        otx_zero_shot_visual_prompting_lit_module,
        fxt_zero_shot_vpm_data_entity,
    ) -> None:
        """Test _inference_step."""
        otx_zero_shot_visual_prompting_lit_module.configure_metric()
        otx_zero_shot_visual_prompting_lit_module.model.return_value = fxt_zero_shot_vpm_data_entity[2]
        mocker_updates = {}
        for k, v in otx_zero_shot_visual_prompting_lit_module.test_metric.items():
            mocker_updates[k] = mocker.patch.object(v, "update")

        otx_zero_shot_visual_prompting_lit_module._inference_step(
            otx_zero_shot_visual_prompting_lit_module.test_metric,
            fxt_zero_shot_vpm_data_entity[1],
            0,
        )

        for v in mocker_updates.values():
            v.assert_called_once()

    def test_inference_step_with_more_preds(
        self,
        mocker,
        otx_zero_shot_visual_prompting_lit_module,
        fxt_zero_shot_vpm_data_entity,
    ) -> None:
        """Test _inference_step with more predictions."""
        otx_zero_shot_visual_prompting_lit_module.configure_metric()
        preds = {}
        for k, v in fxt_zero_shot_vpm_data_entity[2].__dict__.items():
            if k in ["batch_size", "polygons"]:
                preds[k] = v
            else:
                preds[k] = v * 2
        otx_zero_shot_visual_prompting_lit_module.model.return_value = ZeroShotVisualPromptingBatchPredEntity(**preds)
        mocker_updates = {}
        for k, v in otx_zero_shot_visual_prompting_lit_module.test_metric.items():
            mocker_updates[k] = mocker.patch.object(v, "update")

        otx_zero_shot_visual_prompting_lit_module._inference_step(
            otx_zero_shot_visual_prompting_lit_module.test_metric,
            fxt_zero_shot_vpm_data_entity[1],
            0,
        )

        for v in mocker_updates.values():
            v.assert_called_once()

    def test_inference_step_with_more_target(
        self,
        mocker,
        otx_zero_shot_visual_prompting_lit_module,
        fxt_zero_shot_vpm_data_entity,
    ) -> None:
        """Test _inference_step with more targets."""
        otx_zero_shot_visual_prompting_lit_module.configure_metric()
        otx_zero_shot_visual_prompting_lit_module.model.return_value = fxt_zero_shot_vpm_data_entity[2]
        mocker_updates = {}
        for k, v in otx_zero_shot_visual_prompting_lit_module.test_metric.items():
            mocker_updates[k] = mocker.patch.object(v, "update")

        target = {}
        for k, v in fxt_zero_shot_vpm_data_entity[1].__dict__.items():
            if k in ["batch_size"]:
                target[k] = v
            else:
                target[k] = v * 2
        otx_zero_shot_visual_prompting_lit_module._inference_step(
            otx_zero_shot_visual_prompting_lit_module.test_metric,
            ZeroShotVisualPromptingBatchDataEntity(**target),
            0,
        )

        for v in mocker_updates.values():
            v.assert_called_once()
