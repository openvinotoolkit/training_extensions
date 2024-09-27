# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""OTX keypoint detection perfomance benchmark tests."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase


class TestPerfKeypointDetection(PerfTestBase):
    """Benchmark visual prompting."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="keypoint_detection", name="rtmpose_tiny", category="speed"),
    ]

    DATASET_TEST_CASES: ClassVar = [
        Benchmark.Dataset(
            name="coco_person_keypoint_small",
            path=Path("keypoint_detection/coco_keypoint/small"),
            group="small",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="coco_person_keypoint_medium",
            path=Path("keypoint_detection/coco_keypoint/medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="coco_person_keypoint_large",
            path=Path("keypoint_detection/coco_keypoint/large"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/accuracy", summary="max", compare=">", margin=0.1),
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


class TestPerfKeypointDetectionSingleObj(PerfTestBase):
    """Benchmark visual prompting."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="keypoint_detection", name="rtmpose_tiny_single_obj", category="speed"),
    ]

    DATASET_TEST_CASES: ClassVar = [
        Benchmark.Dataset(
            name="coco_person_keypoint_single_obj_small",
            path=Path("keypoint_detection/coco_keypoint_single_obj/small"),
            group="small",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="coco_person_keypoint_single_obj_medium",
            path=Path("keypoint_detection/coco_keypoint_single_obj/medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="coco_person_keypoint_single_obj_large",
            path=Path("keypoint_detection/coco_keypoint_single_obj/large"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/accuracy", summary="max", compare=">", margin=0.1),
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
