# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX semantic segmentation perfomance benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase


class TestPerfSemanticSegmentation(PerfTestBase):
    """Benchmark semantic segmentation."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="semantic_segmentation", name="litehrnet_18", category="balance"),
        Benchmark.Model(task="semantic_segmentation", name="litehrnet_s", category="speed"),
        Benchmark.Model(task="semantic_segmentation", name="litehrnet_x", category="accuracy"),
        Benchmark.Model(task="semantic_segmentation", name="segnext_b", category="other"),
        Benchmark.Model(task="semantic_segmentation", name="segnext_s", category="other"),
        Benchmark.Model(task="semantic_segmentation", name="segnext_t", category="other"),
        Benchmark.Model(task="semantic_segmentation", name="dino_v2", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"kvasir_small_{idx}",
            path=Path("semantic_seg/kvasir_small") / f"{idx}",
            group="small",
            num_repeat=5,
            extra_overrides={},
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="kvasir_medium",
            path=Path("semantic_seg/kvasir_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="kvasir_large",
            path=Path("semantic_seg/kvasir_large"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/Dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/Dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/Dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/Dice", summary="max", compare=">", margin=0.1),
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


class TestPerfSemanticSegmentationSemiSL(TestPerfSemanticSegmentation):
    """Benchmark semantic segmentation."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="semantic_segmentation", name="litehrnet_18_semisl", category="balance"),
        Benchmark.Model(task="semantic_segmentation", name="litehrnet_s_semisl", category="speed"),
        Benchmark.Model(task="semantic_segmentation", name="litehrnet_x_semisl", category="accuracy"),
        Benchmark.Model(task="semantic_segmentation", name="segnext_b_semisl", category="other"),
        Benchmark.Model(task="semantic_segmentation", name="segnext_s_semisl", category="other"),
        Benchmark.Model(task="semantic_segmentation", name="segnext_t_semisl", category="other"),
        Benchmark.Model(task="semantic_segmentation", name="dino_v2_semisl", category="other"),
    ]

    DATASET_TEST_CASES = [  # noqa: RUF012
        Benchmark.Dataset(
            name="kvasir",
            path=Path("semantic_seg/semisl/kvasir_24"),
            group="small",
            num_repeat=5,
            unlabeled_data_path=Path("semantic_seg/semisl/unlabeled_images/kvasir"),
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="kitti",
            path=Path("semantic_seg/semisl/kitti_18"),
            group="small",
            num_repeat=5,
            unlabeled_data_path=Path("semantic_seg/semisl/unlabeled_images/kitti"),
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="cityscapes",
            path=Path("semantic_seg/semisl/cityscapes"),
            group="medium",
            num_repeat=5,
            unlabeled_data_path=Path("semantic_seg/semisl/unlabeled_images/cityscapes"),
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="pascal_voc",
            path=Path("semantic_seg/semisl/pascal_voc"),
            group="large",
            num_repeat=5,
            unlabeled_data_path=Path("semantic_seg/semisl/unlabeled_images/pascal_voc"),
            extra_overrides={},
        ),
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
