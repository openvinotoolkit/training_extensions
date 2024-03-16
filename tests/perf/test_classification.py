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
        Benchmark.Model(task="classification/multi_class_cls", name="efficientnet_b0_light", category="speed"),
        Benchmark.Model(task="classification/multi_class_cls", name="efficientnet_v2_light", category="balance"),
        Benchmark.Model(task="classification/multi_class_cls", name="mobilenet_v3_large_light", category="accuracy"),
        Benchmark.Model(task="classification/multi_class_cls", name="otx_deit_tiny", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="otx_dino_v2", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"multiclass_CUB_small_{idx}",
            path=Path("classification/single_label/multiclass_CUB_small") / f"{idx}",
            size="small",
            data_format="imagenet_with_subset_dirs",
            num_classes=2,
            num_repeat=1,
            extra_overrides={},
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="multiclass_CUB_medium",
            path=Path("classification/single_label/multiclass_CUB_medium"),
            size="medium",
            data_format="imagenet_with_subset_dirs",
            num_classes=67,
            num_repeat=1,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_food101_large",
            path=Path("classification/single_label/multiclass_food101_large"),
            size="large",
            data_format="imagenet_with_subset_dirs",
            num_classes=20,
            num_repeat=1,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="cars",
            path=Path("car_data/car_data"),
            size="large",
            data_format="imagenet_with_subset_dirs",
            num_classes=196,
            num_repeat=1,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="flowers",
            path=Path("flowers"),
            size="large",
            data_format="imagenet_with_subset_dirs",
            num_classes=102,
            num_repeat=1,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="skin",
            path=Path("skin"),
            size="large",
            data_format="imagenet_with_subset_dirs",
            num_classes=14,
            num_repeat=1,
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


class TestPerfMultiLabelClassification(PerfTestBase):
    """Benchmark multi-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/multi_label_cls", name="efficientnet_b0_light", category="speed"),
        Benchmark.Model(task="classification/multi_label_cls", name="efficientnet_v2_light", category="balance"),
        Benchmark.Model(task="classification/multi_label_cls", name="mobilenet_v3_large_light", category="accuracy"),
        Benchmark.Model(task="classification/multi_label_cls", name="otx_deit_tiny", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"multilabel_CUB_small_{idx}",
            path=Path("classification/multi_label/multilabel_CUB_small") / f"{idx}",
            size="small",
            data_format="datumaro",
            num_classes=3,
            num_repeat=1,
            extra_overrides={},
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="multilabel_CUB_medium",
            path=Path("classification/multi_label/multilabel_CUB_medium"),
            size="medium",
            data_format="datumaro",
            num_classes=68,
            num_repeat=1,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multilabel_food101_large",
            path=Path("classification/multi_label/multilabel_food101_large"),
            size="large",
            data_format="datumaro",
            num_classes=21,
            num_repeat=1,
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


class TestPerfHierarchicalLabelClassification(PerfTestBase):
    """Benchmark hierarchical-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/h_label_cls", name="efficientnet_b0_light", category="speed"),
        Benchmark.Model(task="classification/h_label_cls", name="efficientnet_v2_light", category="balance"),
        Benchmark.Model(task="classification/h_label_cls", name="mobilenet_v3_large_light", category="accuracy"),
        Benchmark.Model(task="classification/h_label_cls", name="otx_deit_tiny", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"hlabel_CUB_small_{idx}",
            path=Path("classification/h_label/hlabel_CUB_small") / f"{idx}",
            size="small",
            data_format="datumaro",
            num_classes=6,
            num_repeat=1,
            extra_overrides={
                "model.num_multiclass_heads": "3",
                "model.num_multilabel_classes": "0",
            },
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="hlabel_CUB_medium",
            path=Path("classification/h_label/hlabel_CUB_medium"),
            size="medium",
            data_format="datumaro",
            num_classes=102,
            num_repeat=1,
            extra_overrides={
                "model.num_multiclass_heads": "23",
                "model.num_multilabel_classes": "0",
            },
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
