"""OTX Classification perfomance tests."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import pytest
from pathlib import Path

from .benchmark import Benchmark
from .conftest import PerfTestBase


class TestPerfSingleLabelClassification(PerfTestBase):
    """Benchmark single-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="single_label_classification", name="efficientnet_b0_light", type="speed",),
        Benchmark.Model(task="single_label_classification", name="efficientnet_v2_light", type="balance",),
        Benchmark.Model(task="single_label_classification", name="mobilenet_v3_large_light", type="accuracy",),
        Benchmark.Model(task="single_label_classification", name="otx_deit_tiny", type="other",),
    ]

    DATASET_TEST_CASES = [  # noqa: RUF012
        Benchmark.Dataset(
            name=f"multiclass_CUB_small_{idx}",
            path=Path("multiclass_classification/multiclass_CUB_small") / f"{idx}",
            size="small",
            data_format="imagenet_with_subset_dirs",
            num_classes=2,
            num_repeat=3,
            extra_overrides={},
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name=f"multiclass_CUB_medium",
            path=Path("multiclass_classification/multiclass_CUB_medium"),
            size="medium",
            data_format="imagenet_with_subset_dirs",
            num_classes=67,
            num_repeat=3,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name=f"multiclass_food101_large",
            path=Path("multiclass_classification/multiclass_food101_large"),
            size="large",
            data_format="imagenet_with_subset_dirs",
            num_classes=20,
            num_repeat=1,
            extra_overrides={},
        )
    ]

    ACCURACY_METRICS = [
        Benchmark.Metric(name="test/accuracy", op=">", margin=0.1),
    ]

    EFFICENCY_METRICS = [
        Benchmark.Metric(name="train/iter_time", op="<", margin=0.1),
        Benchmark.Metric(name="test/iter_time", op="<", margin=0.1),
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
    def test_accuracy(self,
        fxt_model: Benchmark.Model,
        fxt_dataset: Benchmark.Dataset,
        fxt_benchmark: Benchmark,
    ):
        self._test_accuracy(
            model=fxt_model,
            dataset=fxt_dataset,
            metrics=self.ACCURACY_METRICS,
            benchmark=fxt_benchmark,
        )

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
    def test_efficiency(self,
        fxt_model: Benchmark.Model,
        fxt_dataset: Benchmark.Dataset,
        fxt_benchmark: Benchmark,
    ):
        self._test_efficiency(
            model=fxt_model,
            dataset=fxt_dataset,
            metrics=self.EFFICENCY_METRICS,
            benchmark=fxt_benchmark,
        )


class TestPerfMultiLabelClassification(PerfTestBase):
    """Benchmark multi-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="multi_label_classification", name="efficientnet_b0_light", type="speed",),
        Benchmark.Model(task="multi_label_classification", name="efficientnet_v2_light", type="balance",),
        Benchmark.Model(task="multi_label_classification", name="mobilenet_v3_large_light", type="accuracy",),
        Benchmark.Model(task="multi_label_classification", name="otx_deit_tiny", type="other",),
    ]

    DATASET_TEST_CASES = [  # noqa: RUF012
        Benchmark.Dataset(
            name=f"multilabel_CUB_small_{idx}",
            path=Path("multilabel_classification/multilabel_CUB_small") / f"{idx}",
            size="small",
            data_format="datumaro",
            num_classes=3,
            num_repeat=3,
            extra_overrides={},
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name=f"multilabel_CUB_medium",
            path=Path("multilabel_classification/multilabel_CUB_medium"),
            size="medium",
            data_format="datumaro",
            num_classes=68,
            num_repeat=3,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name=f"multilabel_food101_large",
            path=Path("multilabel_classification/multilabel_food101_large"),
            size="large",
            data_format="datumaro",
            num_classes=21,
            num_repeat=1,
            extra_overrides={},
        )
    ]

    ACCURACY_METRICS = [
        Benchmark.Metric(name="test/accuracy", op=">", margin=0.1),
    ]

    EFFICENCY_METRICS = [
        Benchmark.Metric(name="train/iter_time", op="<", margin=0.1),
        Benchmark.Metric(name="test/iter_time", op="<", margin=0.1),
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
    def test_accuracy(self,
        fxt_model: Benchmark.Model,
        fxt_dataset: Benchmark.Dataset,
        fxt_benchmark: Benchmark,
    ):
        self._test_accuracy(
            model=fxt_model,
            dataset=fxt_dataset,
            metrics=self.ACCURACY_METRICS,
            benchmark=fxt_benchmark,
        )

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
    def test_efficiency(self,
        fxt_model: Benchmark.Model,
        fxt_dataset: Benchmark.Dataset,
        fxt_benchmark: Benchmark,
    ):
        self._test_efficiency(
            model=fxt_model,
            dataset=fxt_dataset,
            metrics=self.EFFICENCY_METRICS,
            benchmark=fxt_benchmark,
        )


class TestPerfHierarchicalLabelClassification(PerfTestBase):
    """Benchmark hierarchical-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="hierarchical_label_classification", name="efficientnet_b0_light", type="speed",),
        Benchmark.Model(task="hierarchical_label_classification", name="efficientnet_v2_light", type="balance",),
        Benchmark.Model(task="hierarchical_label_classification", name="otx_mobilenet_v3_large_light", type="accuracy",),
        Benchmark.Model(task="hierarchical_label_classification", name="otx_deit_tiny", type="other",),
    ]

    DATASET_TEST_CASES = [  # noqa: RUF012
        Benchmark.Dataset(
            name=f"hlabel_CUB_small_{idx}",
            path=Path("hlabel_classification/hlabel_CUB_small") / f"{idx}",
            size="small",
            data_format="datumaro",
            num_classes=6,
            num_repeat=3,
            extra_overrides={
                "model.num_multiclass_heads": "3",
                "model.num_multilabel_classes": "0",
            },
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name=f"hlabel_CUB_medium",
            path=Path("hlabel_classification/hlabel_CUB_medium"),
            size="medium",
            data_format="datumaro",
            num_classes=102,
            num_repeat=3,
            extra_overrides={
                "model.num_multiclass_heads": "23",
                "model.num_multilabel_classes": "0",
            },
        ),
        # TODO: Add large dataset
    ]

    ACCURACY_METRICS = [
        Benchmark.Metric(name="test/accuracy", op=">", margin=0.1),
    ]

    EFFICENCY_METRICS = [
        Benchmark.Metric(name="train/iter_time", op="<", margin=0.1),
        Benchmark.Metric(name="test/iter_time", op="<", margin=0.1),
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
    def test_accuracy(self,
        fxt_model: Benchmark.Model,
        fxt_dataset: Benchmark.Dataset,
        fxt_benchmark: Benchmark,
    ):
        self._test_accuracy(
            model=fxt_model,
            dataset=fxt_dataset,
            metrics=self.ACCURACY_METRICS,
            benchmark=fxt_benchmark,
        )

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
    def test_efficiency(self,
        fxt_model: Benchmark.Model,
        fxt_dataset: Benchmark.Dataset,
        fxt_benchmark: Benchmark,
    ):
        self._test_efficiency(
            model=fxt_model,
            dataset=fxt_dataset,
            metrics=self.EFFICENCY_METRICS,
            benchmark=fxt_benchmark,
        )
