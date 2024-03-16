# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX object detection perfomance benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase


class TestPerfObjectDetection(PerfTestBase):
    """Benchmark object detection."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="detection", name="atss_mobilenetv2", category="accuracy"),
        Benchmark.Model(task="detection", name="atss_resnext101", category="other"),
        Benchmark.Model(task="detection", name="ssd_mobilenetv2", category="balance"),
        Benchmark.Model(task="detection", name="yolox_tiny", category="speed"),
        Benchmark.Model(task="detection", name="yolox_s", category="other"),
        Benchmark.Model(task="detection", name="yolox_l", category="other"),
        Benchmark.Model(task="detection", name="yolox_x", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"pothole_small_{idx}",
            path=Path("detection/pothole_small") / f"{idx}",
            size="small",
            data_format="coco",
            num_classes=1,
            num_repeat=1,
            extra_overrides={
                # "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasure",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="pothole_medium",
            path=Path("detection/pothole_medium"),
            size="medium",
            data_format="coco",
            num_classes=1,
            num_repeat=1,
            extra_overrides={
                # "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasure",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
        ),
        Benchmark.Dataset(
            name="vitens_large",
            path=Path("detection/vitens_large"),
            size="large",
            data_format="coco",
            num_classes=1,
            num_repeat=1,
            extra_overrides={
                # "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasure",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
        ),
        Benchmark.Dataset(
            name="pascal_tiny",
            path=Path("pascal_tiny/coco_otx"),
            size="large",
            data_format="coco",
            num_classes=20,
            num_repeat=1,
            extra_overrides={
                # "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasure",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/f1-score", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/f1-score", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/f1-score", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/f1-score", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
    ]

    @pytest.mark.parametrize(
        "fxt_model",
        MODEL_TEST_CASES,
        ids=lambda model: model.name,
        indirect=True,
    )
    @pytest.mark.parametrize(
        "fxt_dataset",
        DATASET_TEST_CASES,
        ids=lambda dataset: dataset.name,
        indirect=True,
    )
    def test_perf(
        self,
        fxt_model: Benchmark.Model,
        fxt_dataset: Benchmark.Dataset,
        fxt_benchmark: Benchmark,
    ):
        fxt_benchmark.accelerator = "xpu"
        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
        )
