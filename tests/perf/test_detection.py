"""OTX object detection perfomance benchmark tests."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


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
            num_repeat=3,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.algo.metrices.fmeasure.FMeasure",
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
            num_repeat=3,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.algo.metrices.fmeasure.FMeasure",
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
                "deterministic": "True",
                "metric": "otx.algo.metrices.fmeasure.FMeasure",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
        ),
    ]

    BENCHMARK_TEST_CASES = [  # noqa: RUF012
        {
            "type": "accuracy",
            "criteria": [
                Benchmark.Criterion(name="epoch", summary="max", compare="<", margin=0.1),
                Benchmark.Criterion(name="val/f1-score", summary="max", compare=">", margin=0.1),
                Benchmark.Criterion(name="test/f1-score", summary="max", compare=">", margin=0.1),
                Benchmark.Criterion(name="export/f1-score", summary="max", compare=">", margin=0.1),
                Benchmark.Criterion(name="optimize/f1-score", summary="max", compare=">", margin=0.1),
            ],
        },
        {
            "type": "efficiency",
            "criteria": [
                Benchmark.Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
                Benchmark.Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
                Benchmark.Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
                Benchmark.Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
            ],
        },
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
    @pytest.mark.parametrize(
        "fxt_benchmark",
        BENCHMARK_TEST_CASES,
        ids=lambda benchmark: benchmark["type"],
        indirect=True,
    )
    def test_perf(
        self,
        fxt_model: Benchmark.Model,
        fxt_dataset: Benchmark.Dataset,
        fxt_benchmark: Benchmark,
    ):
        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
        )
