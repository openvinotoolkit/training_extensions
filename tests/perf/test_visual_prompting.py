"""OTX visual prompting perfomance benchmark tests."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from pathlib import Path

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase


class TestPerfVisualPrompting(PerfTestBase):
    """Benchmark visual prompting."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="visual_prompting", name="sam_tiny_vit", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"wgisd_small_{idx}",
            path=Path("visual_prompting/wgisd_small") / f"{idx}",
            size="small",
            data_format="coco",
            num_classes=5,
            num_repeat=3,
            extra_overrides={},
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="coco_car_person_medium",
            path=Path("visual_prompting/coco_car_person_medium"),
            size="medium",
            data_format="coco",
            num_classes=2,
            num_repeat=3,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="vitens_coliform",
            path=Path("visual_prompting/Vitens-Coliform-coco"),
            size="large",
            data_format="coco",
            num_classes=1,
            num_repeat=1,
            extra_overrides={},
        ),
    ]

    BENCHMARK_TEST_CASES = [  # noqa: RUF012
        {
            "type": "accuracy",
            "criteria": [
                Benchmark.Criterion(name="epoch", summary="max", compare="<", margin=0.1),
                Benchmark.Criterion(name="val/Dice", summary="max", compare=">", margin=0.1),
                Benchmark.Criterion(name="test/Dice", summary="max", compare=">", margin=0.1),
                Benchmark.Criterion(name="export/Dice", summary="max", compare=">", margin=0.1),
                Benchmark.Criterion(name="optimize/Dice", summary="max", compare=">", margin=0.1),
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


class TestPerfZeroShotVisualPrompting(PerfTestBase):
    """Benchmark zero-shot visual prompting."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="zero_shot_visual_prompting", name="sam_tiny_vit", category="other"),
        Benchmark.Model(task="zero_shot_visual_prompting", name="sam_vit_b", category="other"),
    ]

    DATASET_TEST_CASES = [  # noqa: RUF012
        Benchmark.Dataset(
            name="coco_car_person_medium_datumaro",
            path=Path("zero_shot_visual_prompting/coco_car_person_medium_datumaro"),
            size="medium",
            data_format="datumaro",
            num_classes=2,
            num_repeat=3,
            extra_overrides={"max_epochs": "1"},
        ),
    ]

    BENCHMARK_TEST_CASES = [  # noqa: RUF012
        {
            "type": "accuracy",
            "criteria": [
                Benchmark.Criterion(name="epoch", summary="max", compare="<", margin=0.1),
                Benchmark.Criterion(name="val/Dice", summary="max", compare=">", margin=0.1),
                Benchmark.Criterion(name="test/Dice", summary="max", compare=">", margin=0.1),
                Benchmark.Criterion(name="export/Dice", summary="max", compare=">", margin=0.1),
                Benchmark.Criterion(name="optimize/Dice", summary="max", compare=">", margin=0.1),
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
