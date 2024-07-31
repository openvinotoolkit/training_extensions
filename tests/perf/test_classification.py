# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX classification perfomance benchmark tests."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def fxt_deterministic(request: pytest.FixtureRequest) -> bool:
    """Override the deterministic setting for classification tasks."""
    deterministic = request.config.getoption("--deterministic")
    deterministic = True if deterministic is None else deterministic == "true"
    log.info(f"{deterministic=}")
    return deterministic


class TestPerfSingleLabelClassification(PerfTestBase):
    """Benchmark single-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/multi_class_cls", name="efficientnet_b0", category="speed"),
        Benchmark.Model(task="classification/multi_class_cls", name="efficientnet_v2", category="balance"),
        Benchmark.Model(task="classification/multi_class_cls", name="mobilenet_v3_large", category="accuracy"),
        Benchmark.Model(task="classification/multi_class_cls", name="deit_tiny", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="dino_v2", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="tv_efficientnet_b3", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="tv_efficientnet_v2_l", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="tv_mobilenet_v3_small", category="other"),
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
    ):
        if fxt_model.name == "dino_v2" and fxt_accelerator == "xpu":
            pytest.skip(f"{fxt_model.name} doesn't support {fxt_accelerator}.")

        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
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
    ):
        if fxt_model.name == "dino_v2" and fxt_accelerator == "xpu":
            pytest.skip(f"{fxt_model.name} doesn't support {fxt_accelerator}.")

        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
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
    ):
        if fxt_model.name == "dino_v2" and fxt_accelerator == "xpu":
            pytest.skip(f"{fxt_model.name} doesn't support {fxt_accelerator}.")

        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
        )


class TestPerfSemiSLMultiClass(PerfTestBase):
    """Benchmark single-label classification for Semi-SL task."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/multi_class_cls", name="efficientnet_b0_semisl", category="balance"),
        Benchmark.Model(task="classification/multi_class_cls", name="mobilenet_v3_large_semisl", category="speed"),
        Benchmark.Model(task="classification/multi_class_cls", name="efficientnet_v2_semisl", category="accuracy"),
        Benchmark.Model(task="classification/multi_class_cls", name="deit_tiny_semisl", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="dino_v2_semisl", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="tv_efficientnet_b3_semisl", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="tv_efficientnet_v2_l_semisl", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="tv_mobilenet_v3_small_semisl", category="other"),
    ]

    DATASET_TEST_CASES = (
        [
            Benchmark.Dataset(
                name=f"cifar10@{num_label}_{idx}",
                path=Path(f"multiclass_classification/semi-sl/cifar10@{num_label}_{idx}/supervised"),
                group="cifar10",
                num_repeat=1,
                unlabeled_data_path=Path(f"multiclass_classification/semi-sl/cifar10@{num_label}_{idx}/unlabel_data"),
                extra_overrides={
                    "train": {
                        "data.train_subset.subset_name": "train_data",
                        "data.val_subset.subset_name": "val_data",
                        "data.test_subset.subset_name": "val_data",
                    },
                },
            )
            for idx in (0, 1, 2)
            for num_label in (4, 10, 25)
        ]
        + [
            Benchmark.Dataset(
                name=f"svhn@{num_label}_{idx}",
                path=Path(f"multiclass_classification/semi-sl/svhn@{num_label}_{idx}/supervised"),
                group="svhn",
                num_repeat=1,
                unlabeled_data_path=Path(f"multiclass_classification/semi-sl/svhn@{num_label}_{idx}/unlabel_data"),
                extra_overrides={
                    "train": {
                        "data.train_subset.subset_name": "train_data",
                        "data.val_subset.subset_name": "val_data",
                        "data.test_subset.subset_name": "val_data",
                    },
                },
            )
            for idx in (0, 1, 2)
            for num_label in (4, 10, 25)
        ]
        + [
            Benchmark.Dataset(
                name=f"fmnist@{num_label}_{idx}",
                path=Path(f"multiclass_classification/semi-sl/fmnist@{num_label}_{idx}/supervised"),
                group="fmnist",
                num_repeat=1,
                unlabeled_data_path=Path(f"multiclass_classification/semi-sl/fmnist@{num_label}_{idx}/unlabel_data"),
                extra_overrides={
                    "train": {
                        "data.train_subset.subset_name": "train_data",
                        "data.val_subset.subset_name": "val_data",
                        "data.test_subset.subset_name": "val_data",
                    },
                },
            )
            for idx in (0, 1, 2)
            for num_label in (4, 10, 25)
        ]
    )

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
