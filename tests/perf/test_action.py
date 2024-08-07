# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX action perfomance benchmark tests."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def fxt_deterministic(request: pytest.FixtureRequest) -> bool:
    """Override the deterministic setting for action classification task."""
    deterministic = request.config.getoption("--deterministic")
    deterministic = False if deterministic is None else deterministic == "true"
    log.info(f"{deterministic=}")
    return deterministic


class TestPerfActionClassification(PerfTestBase):
    """Benchmark action classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="action_classification", name="movinet", category="speed"),
        Benchmark.Model(task="action_classification", name="x3d", category="accuracy"),
    ]

    DATASET_TEST_CASES = [  # noqa: RUF012
        Benchmark.Dataset(
            name="ucf-5percent-small",
            path=Path("action/action_classification/ucf_kinetics_5percent_small"),
            group="small",
            num_repeat=5,
            extra_overrides={
                "train": {
                    "max_epochs": "10",
                },
            },
        ),
        Benchmark.Dataset(
            name="ucf-30percent-medium",
            path=Path("action/action_classification/ucf_kinetics_30percent_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={
                "train": {
                    "max_epochs": "10",
                },
            },
        ),
        Benchmark.Dataset(
            name="ucf-large",
            path=Path("action/action_classification/ucf_kinetics_large"),
            group="large",
            num_repeat=5,
            extra_overrides={
                "train": {
                    "max_epochs": "3",
                },
            },
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="test/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/test/accuracy", summary="max", compare=">", margin=0.1),
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
