# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import CSVLogger

from getitune.backend.lightning.callbacks.gpu_mem_monitor import GPUMemMonitor


class TestGPUMemMonitor:
    def test_gpu_monitor(self, tmpdir):
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")

        monitor = GPUMemMonitor()
        model = BoringModel()

        class DebugLogger(CSVLogger):
            def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
                assert "gpu_mem_allocated_gib" in metrics
                assert "gpu_mem_reserved_gib" in metrics

        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=2,
            log_every_n_steps=1,
            accelerator="gpu",
            devices=1,
            callbacks=[monitor],
            logger=DebugLogger(tmpdir),
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        trainer.fit(model)
