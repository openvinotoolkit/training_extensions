# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX instance segmentation perfomance benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase


class TestPerfInstanceSegmentation(PerfTestBase):
    """Benchmark instance segmentation."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="instance_segmentation", name="maskrcnn_efficientnetb2b", category="speed"),
        Benchmark.Model(task="instance_segmentation", name="maskrcnn_r50", category="accuracy"),
        Benchmark.Model(task="instance_segmentation", name="maskrcnn_swint", category="other"),
        Benchmark.Model(task="instance_segmentation", name="rtmdet_inst_tiny", category="other"),
        Benchmark.Model(task="instance_segmentation", name="maskrcnn_r50_tv", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"wgisd_small_{idx}",
            path=Path("instance_seg/wgisd_small") / f"{idx}",
            group="small",
            num_repeat=5,
            extra_overrides={
                "train": {
                    "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                    "callback_monitor": "val/f1-score",
                    "model.scheduler.monitor": "val/f1-score",
                },
                "test": {
                    "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                },
            },
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="coco_car_person_medium",
            path=Path("instance_seg/coco_car_person_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={
                "train": {
                    "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                    "callback_monitor": "val/f1-score",
                    "model.scheduler.monitor": "val/f1-score",
                },
                "test": {
                    "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                },
            },
        ),
        Benchmark.Dataset(
            name="vitens_coliform",
            path=Path("instance_seg/Vitens-Coliform-coco"),
            group="large",
            num_repeat=5,
            extra_overrides={
                "train": {
                    "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                    "callback_monitor": "val/f1-score",
                    "model.scheduler.monitor": "val/f1-score",
                },
                "test": {
                    "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                },
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
        Benchmark.Criterion(name="test(train)/e2e_time", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
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
        fxt_accelerator: str,
    ):
        if fxt_model.name == "maskrcnn_r50" and fxt_accelerator == "xpu":
            pytest.skip(f"{fxt_model.name} doesn't support {fxt_accelerator}.")

        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
        )


class TestPerfTilingInstanceSegmentation(PerfTestBase):
    """Benchmark tiling instance segmentation."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="instance_segmentation", name="maskrcnn_efficientnetb2b_tile", category="speed"),
        Benchmark.Model(task="instance_segmentation", name="maskrcnn_r50_tile", category="accuracy"),
        Benchmark.Model(task="instance_segmentation", name="maskrcnn_swint_tile", category="other"),
        Benchmark.Model(task="instance_segmentation", name="rtmdet_inst_tiny_tile", category="other"),
        Benchmark.Model(task="instance_segmentation", name="maskrcnn_r50_tv_tile", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"vitens_aeromonas_small_{idx}",
            path=Path("tiling_instance_seg/vitens_aeromonas_small") / f"{idx}",
            group="small",
            num_repeat=5,
            extra_overrides={
                "train": {
                    "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                    "callback_monitor": "val/f1-score",
                    "model.scheduler.monitor": "val/f1-score",
                },
                "test": {
                    "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                },
            },
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="vitens_aeromonas_medium",
            path=Path("tiling_instance_seg/vitens_aeromonas_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={
                "train": {
                    "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                    "callback_monitor": "val/f1-score",
                    "model.scheduler.monitor": "val/f1-score",
                },
                "test": {
                    "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                },
            },
        ),
        # Add large dataset
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
        Benchmark.Criterion(name="test(train)/e2e_time", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
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
        fxt_accelerator: str,
    ):
        if fxt_model.name == "maskrcnn_r50" and fxt_accelerator == "xpu":
            pytest.skip(f"{fxt_model.name} doesn't support {fxt_accelerator}.")

        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
        )
