# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for detection model module."""

from __future__ import annotations

from functools import partial
from unittest.mock import create_autospec

import pytest
from lightning.pytorch.cli import ReduceLROnPlateau
from torch.optim import Optimizer

from otx.algo.schedulers.warmup_schedulers import LinearWarmupScheduler
from otx.core.metrics.fmeasure import FMeasure
from otx.core.model.entity.detection import OTXDetectionModel
from otx.core.model.module.base import OTXLitModule
from otx.core.model.module.detection import OTXDetectionLitModule


class TestOTXLitModule:
    @pytest.fixture()
    def mock_otx_model(self) -> OTXDetectionModel:
        return create_autospec(OTXDetectionModel)

    @pytest.fixture()
    def mock_optimizer(self) -> Optimizer:
        return create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self) -> list[LinearWarmupScheduler | ReduceLROnPlateau]:
        return create_autospec([LinearWarmupScheduler, ReduceLROnPlateau])

    def test_configure_metric_with_v1_ckpt(
        self,
        mock_otx_model,
        mock_optimizer,
        mock_scheduler,
        mocker,
    ) -> None:
        mock_otx_model.test_meta_info = {}
        module = OTXDetectionLitModule(
            otx_model=mock_otx_model,
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            metric=partial(FMeasure),
        )

        mock_v1_ckpt = {
            "confidence_threshold": 0.35,
            "state_dict": {},
        }

        mocker.patch.object(OTXLitModule, "load_state_dict", return_value=None)
        module.load_state_dict(mock_v1_ckpt)

        assert module.test_meta_info["best_confidence_threshold"] == 0.35
        assert module.model.test_meta_info["best_confidence_threshold"] == 0.35

        module.configure_metric()
        assert module.metric.best_confidence_threshold == 0.35

    def test_configure_metric_with_v2_ckpt(
        self,
        mock_otx_model,
        mock_optimizer,
        mock_scheduler,
        mocker,
    ) -> None:
        mock_otx_model.test_meta_info = {}
        module = OTXDetectionLitModule(
            otx_model=mock_otx_model,
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            metric=partial(FMeasure),
        )

        mock_v2_ckpt = {
            "hyper_parameters": {"confidence_threshold": 0.35},
            "state_dict": {},
        }

        mocker.patch.object(OTXLitModule, "load_state_dict", return_value=None)
        module.load_state_dict(mock_v2_ckpt)

        assert module.test_meta_info["best_confidence_threshold"] == 0.35
        assert module.model.test_meta_info["best_confidence_threshold"] == 0.35

        module.configure_metric()
        assert module.metric.best_confidence_threshold == 0.35
