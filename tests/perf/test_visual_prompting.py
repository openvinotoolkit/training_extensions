# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX visual prompting perfomance benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase


class TestPerfVisualPrompting(PerfTestBase):
    """Benchmark visual prompting."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="visual_prompting", name="sam_tiny_vit", category="speed"),
        Benchmark.Model(task="visual_prompting", name="sam_vit_b", category="accuracy"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"wgisd_small_{idx}",
            path=Path("visual_prompting/wgisd_small") / f"{idx}",
            group="small",
            num_repeat=5,
            extra_overrides={},
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="coco_car_person_medium",
            path=Path("visual_prompting/coco_car_person_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="vitens_coliform",
            path=Path("visual_prompting/Vitens-Coliform-coco"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/dice", summary="max", compare=">", margin=0.1),
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


class TestPerfZeroShotVisualPrompting(PerfTestBase):
    """Benchmark zero-shot visual prompting."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="zero_shot_visual_prompting", name="sam_tiny_vit", category="speed"),
        Benchmark.Model(task="zero_shot_visual_prompting", name="sam_vit_b", category="accuracy"),
    ]

    DATASET_TEST_CASES = [  # noqa: RUF012
        Benchmark.Dataset(
            name="coco_car_person_medium",
            path=Path("zero_shot_visual_prompting/coco_car_person_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={
                "train": {
                    "max_epochs": "1",
                },
            },
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/dice", summary="max", compare=">", margin=0.1),
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
