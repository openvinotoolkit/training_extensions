# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for torch.compile compatibility of model architectures.

These tests verify that models can be compiled with torch.compile without errors.
"""

from __future__ import annotations

import gc

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.detection.atss import ATSS
from getitune.backend.lightning.models.detection.rtdetr import RTDETR
from getitune.backend.lightning.models.detection.yolox import YOLOX
from getitune.backend.lightning.models.segmentation.dino_v2_seg import DinoV2Seg
from getitune.backend.lightning.models.segmentation.segnext import SegNext
from getitune.config import register_configs


@pytest.fixture(scope="session", autouse=True)
def fxt_register_configs() -> None:
    register_configs()


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Cleanup memory after each test to prevent OOM errors."""
    yield
    gc.collect()
    torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_compile_test(model: torch.nn.Module, input_size: tuple[int, int]) -> None:
    """Helper to run a torch.compile test on a model."""
    torch._dynamo.reset()
    cnt = CompileCounter()
    model.model = torch.compile(model.model, backend=cnt)  # pyrefly: ignore[no-matching-overload]
    x = torch.randn(1, 3, *input_size)
    model.model(x)  # pyrefly: ignore[bad-argument-type, not-callable]
    assert cnt.frame_count == 1


class TestCompiledModelsDetection:
    """Integration tests for torch.compile on detection models."""

    @pytest.mark.parametrize("model_name", ["atss_mobilenetv2", "atss_resnext101"])
    def test_atss_compiled(self, model_name: str) -> None:
        model = ATSS(
            model_name=model_name,  # pyrefly: ignore[bad-argument-type]
            label_info=3,
            data_input_params=DataInputParams((800, 992), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        _run_compile_test(model, model.data_input_params.input_size)

    @pytest.mark.parametrize("model_name", ["yolox_tiny", "yolox_s", "yolox_l", "yolox_x"])
    def test_yolox_compiled(self, model_name: str) -> None:
        model = YOLOX(
            model_name=model_name,  # pyrefly: ignore[bad-argument-type]
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        _run_compile_test(model, model.data_input_params.input_size)

    @pytest.mark.parametrize("model_name", ["rtdetr_50"])
    def test_rtdetr_compiled(self, model_name: str) -> None:
        model = RTDETR(
            model_name=model_name,  # pyrefly: ignore[bad-argument-type]
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        model.model.training = False  # do not calculate loss
        _run_compile_test(model, model.data_input_params.input_size)

    def test_ssd_compiled(self) -> None:
        from getitune.backend.lightning.models.detection import SSD

        model = SSD(
            model_name="ssd_mobilenetv2",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        _run_compile_test(model, model.data_input_params.input_size)


class TestCompiledModelsSegmentation:
    """Integration tests for torch.compile on segmentation models."""

    def test_dinov2_seg_compiled(self) -> None:
        model = DinoV2Seg(
            model_name="dinov2-small-seg",
            label_info=3,
            data_input_params=DataInputParams((518, 518), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        _run_compile_test(model, model.data_input_params.input_size)

    @pytest.mark.parametrize("model_name", ["segnext_tiny", "segnext_small", "segnext_base"])
    def test_segnext_compiled(self, model_name: str) -> None:
        model = SegNext(
            model_name=model_name,  # pyrefly: ignore[bad-argument-type]
            label_info=3,
            data_input_params=DataInputParams((518, 518), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        _run_compile_test(model, model.data_input_params.input_size)
