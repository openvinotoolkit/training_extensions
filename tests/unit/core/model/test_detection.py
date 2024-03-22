# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for detection model module."""

from __future__ import annotations

from functools import partial
from unittest.mock import create_autospec

import pytest
from lightning.pytorch.cli import ReduceLROnPlateau
from otx.algo.schedulers.warmup_schedulers import LinearWarmupScheduler
from otx.core.metrics.fmeasure import FMeasure
from otx.core.model.detection import OTXDetectionModel
from torch.optim import Optimizer


class TestOTXDetectionModel:
    @pytest.fixture()
    def mock_optimizer(self) -> Optimizer:
        return create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self) -> list[LinearWarmupScheduler | ReduceLROnPlateau]:
        return create_autospec([LinearWarmupScheduler, ReduceLROnPlateau])

    @pytest.fixture(
        params=[
            {
                "confidence_threshold": 0.35,
                "state_dict": {},
            },
            {
                "hyper_parameters": {"confidence_threshold": 0.35},
                "state_dict": {},
            },
        ],
        ids=["v1", "v2"],
    )
    def mock_ckpt(self, request):
        return request.param

    def test_configure_metric_with_ckpt(
        self,
        mock_optimizer,
        mock_scheduler,
        mock_ckpt,
    ) -> None:
        model = OTXDetectionModel(
            num_classes=1,
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            metric=partial(FMeasure),
        )

        model.load_state_dict(mock_ckpt)

        assert model.test_meta_info["best_confidence_threshold"] == 0.35

        model.configure_metric()
        assert model.metric.best_confidence_threshold == 0.35
