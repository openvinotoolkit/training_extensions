# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX classification perfomance benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase


class TestPerfSingleLabelClassification(PerfTestBase):
    """Benchmark single-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/multi_class_cls", name="efficientnet_b0", category="speed"),
        Benchmark.Model(task="classification/multi_class_cls", name="efficientnet_v2", category="balance"),
        Benchmark.Model(task="classification/multi_class_cls", name="mobilenet_v3_large", category="accuracy"),
        Benchmark.Model(task="classification/multi_class_cls", name="deit_tiny", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="dino_v2", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"multiclass_CUB_small_{idx}",
            path=Path("multiclass_classification/multiclass_CUB_small") / f"{idx}",
            group="small",
            num_repeat=5,
            extra_overrides={},
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="multiclass_CUB_medium",
            path=Path("multiclass_classification/multiclass_CUB_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_food101_large",
            path=Path("multiclass_classification/multiclass_food101_large"),
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
        fxt_accelerator: str,
        fxt_resume_from: Path | None,
    ):
        if fxt_model.name == "dino_v2" and fxt_accelerator == "xpu":
            pytest.skip(f"{fxt_model.name} doesn't support {fxt_accelerator}.")

        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
            resume_from=fxt_resume_from,
        )


class TestPerfMultiLabelClassification(PerfTestBase):
    """Benchmark multi-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/multi_label_cls", name="efficientnet_b0", category="speed"),
        Benchmark.Model(task="classification/multi_label_cls", name="efficientnet_v2", category="balance"),
        Benchmark.Model(task="classification/multi_label_cls", name="mobilenet_v3_large", category="accuracy"),
        Benchmark.Model(task="classification/multi_label_cls", name="deit_tiny", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"multilabel_CUB_small_{idx}",
            path=Path("multilabel_classification/multilabel_CUB_small") / f"{idx}",
            group="small",
            num_repeat=5,
            extra_overrides={},
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="multilabel_CUB_medium",
            path=Path("multilabel_classification/multilabel_CUB_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multilabel_food101_large",
            path=Path("multilabel_classification/multilabel_food101_large"),
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
        fxt_accelerator: str,
        fxt_resume_from: Path | None,
    ):
        if fxt_model.name == "dino_v2" and fxt_accelerator == "xpu":
            pytest.skip(f"{fxt_model.name} doesn't support {fxt_accelerator}.")

        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
            resume_from=fxt_resume_from,
        )


class TestPerfHierarchicalLabelClassification(PerfTestBase):
    """Benchmark hierarchical-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/h_label_cls", name="efficientnet_b0", category="speed"),
        Benchmark.Model(task="classification/h_label_cls", name="efficientnet_v2", category="balance"),
        Benchmark.Model(task="classification/h_label_cls", name="mobilenet_v3_large", category="accuracy"),
        Benchmark.Model(task="classification/h_label_cls", name="deit_tiny", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"hlabel_CUB_small_{idx}",
            path=Path("hlabel_classification/hlabel_CUB_small") / f"{idx}",
            group="small",
            num_repeat=5,
            extra_overrides={},
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="hlabel_CUB_medium",
            path=Path("hlabel_classification/hlabel_CUB_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        # Add large dataset
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
        fxt_accelerator: str,
        fxt_resume_from: Path | None,
    ):
        if fxt_model.name == "dino_v2" and fxt_accelerator == "xpu":
            pytest.skip(f"{fxt_model.name} doesn't support {fxt_accelerator}.")

        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
            resume_from=fxt_resume_from,
        )
