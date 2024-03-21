# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for classification model module."""

from __future__ import annotations

import pytest
import torchmetrics
from lightning.pytorch.cli import ReduceLROnPlateau
from otx.algo.schedulers.warmup_schedulers import LinearWarmupScheduler
from unittest.mock import create_autospec
from torchmetrics import Metric
from otx.core.model.entity.classification import (
    OTXMulticlassClsModel,
    OTXMultilabelClsModel,
    OTXHlabelClsModel
)
from otx.core.model.module.classification import (
    OTXMulticlassClsLitModule,
    OTXMultilabelClsLitModule,
    OTXHlabelClsLitModule
)
from torch.optim import Optimizer
from otx.core.metrics.accuracy import HlabelAccuracy

@pytest.fixture()
def mock_otx_model() -> list[OTXMulticlassClsModel, OTXMultilabelClsModel, OTXHlabelClsModel]:
    return [
        create_autospec(OTXMulticlassClsModel, instance=True),
        create_autospec(OTXMultilabelClsModel, instance=True),
        create_autospec(OTXHlabelClsModel, instance=True)
    ]

@pytest.fixture()
def mock_optimizer() -> Optimizer:
    return create_autospec(Optimizer)

@pytest.fixture()
def mock_scheduler() -> list[LinearWarmupScheduler | ReduceLROnPlateau]:
    return create_autospec([LinearWarmupScheduler, ReduceLROnPlateau])

@pytest.fixture()
def mock_modules(mock_otx_model, mock_optimizer, mock_scheduler) -> list:
    return [
        OTXMulticlassClsLitModule(
            otx_model=mock_otx_model[0],
            torch_compile=False,
            optimizer=mock_optimizer,  
            scheduler=mock_scheduler,  
            metric=torchmetrics.Accuracy(task="multiclass", num_classes=2)
        ),
        OTXMultilabelClsLitModule(
            otx_model=mock_otx_model[1],
            torch_compile=False,
            optimizer=mock_optimizer,  
            scheduler=mock_scheduler,  
            metric=torchmetrics.Accuracy(task="multilabel", num_labels=2)
        ),
        OTXHlabelClsLitModule(
            otx_model=mock_otx_model[1],
            torch_compile=False,
            optimizer=mock_optimizer,  
            scheduler=mock_scheduler,  
            metric=HlabelAccuracy()
        )
    ]
    

class TestOTXClsLitModule:
    def test_configure_metric(self, mock_modules):
        for module in mock_modules:
            module.configure_metric()
            assert isinstance(module.metric, Metric), "Metric should be an instance of Metric."

    def test_validation_step(
        self, 
        mock_modules,
        fxt_multiclass_cls_data_entity,
        fxt_multilabel_cls_data_entity,
        fxt_hlabel_cls_data_entity,
        fxt_hlabel_multilabel_info
    ):
        batch_entity = [
            fxt_multiclass_cls_data_entity[1], fxt_multilabel_cls_data_entity[1], fxt_hlabel_cls_data_entity[1]
        ]
        pred_entity = [
            fxt_multiclass_cls_data_entity[2], fxt_multilabel_cls_data_entity[2], fxt_hlabel_cls_data_entity[2]
        ]
        for idx, module in enumerate(mock_modules):
            module.configure_metric()
            if isinstance(module, OTXHlabelClsLitModule):
                module.label_info = fxt_hlabel_multilabel_info
                module.model.set_hlabel_info.return_value = []
            module.model.return_value = pred_entity[idx]
            module.validation_step(batch_entity[idx], batch_idx=0)
        