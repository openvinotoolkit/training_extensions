# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX anomaly perfomance benchmark tests."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase


class TestPerfAnomalyClassification(PerfTestBase):
    """Benchmark anomaly classification."""

    MODEL_TEST_CASES: ClassVar[list[Benchmark.Model]] = [
        Benchmark.Model(task="anomaly_classification", name="padim", category="speed"),
        Benchmark.Model(task="anomaly_classification", name="stfpm", category="accuracy"),
    ]

    DATASET_TEST_CASES: ClassVar[list[Benchmark.Dataset]] = [
        # TODO(Emily): Need to replace small datasets with appropriate ones.
        # Currently excluding small datasets from benchmark testing until replacements are ready.
        # Small datasets to be replaced include: mvtec_bottle_small_1, mvtec_bottle_small_2, mvtec_bottle_small_3.
        Benchmark.Dataset(
            name="mvtec_wood_medium",
            path=Path("anomaly/mvtec/wood_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="mvtec_hazelnut_large",
            path=Path("anomaly/mvtec/hazelnut_large"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="test/image_F1Score", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/image_F1Score", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/image_F1Score", summary="max", compare=">", margin=0.1),
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
    ):
        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
        )


class TestPerfAnomalyDetection(PerfTestBase):
    """Benchmark anomaly detection."""

    MODEL_TEST_CASES: ClassVar[list[Benchmark.Model]] = [
        Benchmark.Model(task="anomaly_detection", name="padim", category="speed"),
        Benchmark.Model(task="anomaly_detection", name="stfpm", category="accuracy"),
    ]

    DATASET_TEST_CASES: ClassVar[list[Benchmark.Dataset]] = [
        # TODO(Emily): Need to replace small datasets with appropriate ones.
        # Currently excluding small datasets from benchmark testing until replacements are ready.
        # Small datasets to be replaced include: mvtec_bottle_small_1, mvtec_bottle_small_2, mvtec_bottle_small_3.
        Benchmark.Dataset(
            name="mvtec_wood_medium",
            path=Path("anomaly/mvtec/wood_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="mvtec_hazelnut_large",
            path=Path("anomaly/mvtec/hazelnut_large"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="test/image_F1Score", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/image_F1Score", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/image_F1Score", summary="max", compare=">", margin=0.1),
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
    ):
        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
        )


class TestPerfAnomalySegmentation(PerfTestBase):
    """Benchmark anomaly segmentation."""

    MODEL_TEST_CASES: ClassVar[list[Benchmark.Model]] = [
        Benchmark.Model(task="anomaly_segmentation", name="padim", category="speed"),
        Benchmark.Model(task="anomaly_segmentation", name="stfpm", category="accuracy"),
    ]

    DATASET_TEST_CASES: ClassVar[list[Benchmark.Dataset]] = [
        # TODO(Emily): Need to replace small datasets with appropriate ones.
        # Currently excluding small datasets from benchmark testing until replacements are ready.
        # Small datasets to be replaced include: mvtec_bottle_small_1, mvtec_bottle_small_2, mvtec_bottle_small_3.
        Benchmark.Dataset(
            name="mvtec_wood_medium",
            path=Path("anomaly/mvtec/wood_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="mvtec_hazelnut_large",
            path=Path("anomaly/mvtec/hazelnut_large"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="test/pixel_F1Score", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/pixel_F1Score", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/pixel_F1Score", summary="max", compare=">", margin=0.1),
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
    ):
        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
        )
