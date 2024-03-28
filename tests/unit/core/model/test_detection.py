# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for detection model module."""

from __future__ import annotations

from unittest.mock import create_autospec

import pytest
from lightning.pytorch.cli import ReduceLROnPlateau
from otx.core.metrics.fmeasure import FMeasureCallable
from otx.core.model.detection import OTXDetectionModel
from torch.optim import Optimizer


class TestOTXDetectionModel:
    @pytest.fixture()
    def mock_optimizer(self):
        return lambda _: create_autospec(Optimizer)

    @pytest.fixture()
    def mock_scheduler(self):
        return lambda _: create_autospec([ReduceLROnPlateau])

    @pytest.fixture(
        params=[
            {
                "confidence_threshold": 0.35,
                "state_dict": {},
            },
            {
                "hyper_parameters": {"best_confidence_threshold": 0.35},
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
            metric=FMeasureCallable,
        )

        model.load_state_dict(mock_ckpt)

        assert model.hparams["best_confidence_threshold"] == 0.35
