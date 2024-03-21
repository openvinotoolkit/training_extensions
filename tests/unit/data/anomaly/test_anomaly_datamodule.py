"""Tests for anomaly datamodule."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torchvision.transforms.v2 import Normalize, Resize, ToDtype

from otx.core.types.task import OTXTaskType
from otx.data.anomaly import AnomalyDataModule


class TestAnomalyDatamodule:
    @pytest.mark.parametrize(
        "task",
        [OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION],
    )
    def test_datamodule(self, task: OTXTaskType):
        """Test datamodule."""
        transforms = [
            Resize((256, 256), antialias=True),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        datamodule = AnomalyDataModule(
            task_type=task,
            data_dir="tests/assets/anomaly_hazelnut",
            train_transforms=transforms,
            val_transforms=transforms,
            test_transforms=transforms,
            train_batch_size=8,
            val_batch_size=8,
            test_batch_size=8,
        )
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        test_dataloader = datamodule.test_dataloader()
        for dataloader in (train_dataloader, val_dataloader, test_dataloader):
            batch = next(iter(dataloader))
            assert batch.batch_size == 8
            assert batch.images.shape == (8, 3, 256, 256)
            assert batch.labels is not None
            if task == OTXTaskType.ANOMALY_DETECTION:
                assert batch.boxes is not None
            elif task == OTXTaskType.ANOMALY_SEGMENTATION:
                assert batch.masks is not None
