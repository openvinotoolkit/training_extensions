# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from otx.cli.train import otx_train

import mlflow


@dataclass
class ModelTestCase:
    task: str
    name: str


@dataclass
class DatasetTestCase:
    name: str
    data_root: Path
    data_format: str
    num_classes: int
    extra_overrides: dict


@dataclass
class RegressionTestCase:
    model: ModelTestCase
    dataset: DatasetTestCase
    output_dir: Path


class BaseTest:
    def _test_regression(
        self,
        model_test_case: ModelTestCase,
        dataset_test_case: DatasetTestCase,
        fxt_dataset_root_dir: Path,
        fxt_tags: dict,
        fxt_num_repeat: int,
        tmpdir: pytest.TempdirFactory,
    ) -> None:
        for seed in range(fxt_num_repeat):
            test_case = RegressionTestCase(
                model=model_test_case,
                dataset=dataset_test_case,
                output_dir=Path(tmpdir) / str(seed),
            )

            run_name = f"{test_case.model.task}/{test_case.model.name}/{test_case.dataset.name}/{seed}"
            tags = {
                "task": test_case.model.task,
                "model": test_case.model.name,
                "dataset": test_case.dataset.name,
                "seed": str(seed),
                **fxt_tags,
            }
            data_root = (
                fxt_dataset_root_dir
                / test_case.model.task
                / test_case.dataset.data_root
            )
            with mlflow.start_run(tags=tags, run_name=run_name):
                overrides = [
                    f"+recipe={test_case.model.task}/{test_case.model.name}",
                    f"model.otx_model.config.head.num_classes={test_case.dataset.num_classes}",
                    f"data.data_root={data_root}",
                    f"data.data_format={test_case.dataset.data_format}",
                    f"base.output_dir={test_case.output_dir}",
                    f"seed={seed}",
                    "test=true",
                ] + [
                    f"{key}={value}"
                    for key, value in test_case.dataset.extra_overrides.items()
                ]
                _, metrics = otx_train(overrides)

                # Submit metrics to MLFlow Tracker server
                mlflow.log_metrics(metrics)


class TestMultiClassCls(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="classification", name="otx_deit_tiny"),
        ModelTestCase(task="classification", name="otx_dino_v2"),
        ModelTestCase(task="classification", name="otx_efficientnet_b0"),
        ModelTestCase(task="classification", name="otx_efficientnet_v2"),
        ModelTestCase(task="classification", name="otx_mobilenet_v3_large"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name=f"multiclass_CUB_small_{idx}",
            data_root=Path("multiclass_CUB_small") / f"{idx}",
            data_format="imagenet_with_subset_dirs",
            num_classes=2,
            extra_overrides={"trainer.max_epochs": "20"},
        )
        for idx in range(1, 4)
    ]

    @pytest.mark.parametrize(
        "model_test_case",
        MODEL_TEST_CASES,
        ids=[tc.name for tc in MODEL_TEST_CASES],
    )
    @pytest.mark.parametrize(
        "dataset_test_case",
        DATASET_TEST_CASES,
        ids=[tc.name for tc in DATASET_TEST_CASES],
    )
    def test_regression(
        self,
        model_test_case: ModelTestCase,
        dataset_test_case: DatasetTestCase,
        fxt_dataset_root_dir: Path,
        fxt_tags: dict,
        fxt_num_repeat: int,
        tmpdir: pytest.TempdirFactory,
    ) -> None:
        self._test_regression(
            model_test_case=model_test_case,
            dataset_test_case=dataset_test_case,
            fxt_dataset_root_dir=fxt_dataset_root_dir,
            fxt_tags=fxt_tags,
            fxt_num_repeat=fxt_num_repeat,
            tmpdir=tmpdir,
        )


class TestInstanceSegmentation(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="instance_segmentation", name="maskrcnn_efficientnetb2b"),
        ModelTestCase(task="instance_segmentation", name="maskrcnn_r50"),
        ModelTestCase(task="instance_segmentation", name="maskrcnn_swint"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name="wgisd_small/3",
            data_root=Path("wgisd_small/3"),
            data_format="coco_instances",
            num_classes=5,
            extra_overrides={"trainer.max_epochs": "55"},
        ),
        DatasetTestCase(
            name="coco_car_person_medium",
            data_root=Path("coco_car_person_medium"),
            data_format="coco_instances",
            num_classes=2,
            extra_overrides={"trainer.max_epochs": "20"},
        ),
    ]

    @pytest.mark.parametrize(
        "model_test_case",
        MODEL_TEST_CASES,
        ids=[tc.name for tc in MODEL_TEST_CASES],
    )
    @pytest.mark.parametrize(
        "dataset_test_case",
        DATASET_TEST_CASES,
        ids=[tc.name for tc in DATASET_TEST_CASES],
    )
    def test_regression(
        self,
        model_test_case: ModelTestCase,
        dataset_test_case: DatasetTestCase,
        fxt_dataset_root_dir: Path,
        fxt_tags: dict,
        fxt_num_repeat: int,
        tmpdir: pytest.TempdirFactory,
    ) -> None:
        self._test_regression(
            model_test_case=model_test_case,
            dataset_test_case=dataset_test_case,
            fxt_dataset_root_dir=fxt_dataset_root_dir,
            fxt_tags=fxt_tags,
            fxt_num_repeat=fxt_num_repeat,
            tmpdir=tmpdir,
        )

    def _test_regression(
        self,
        model_test_case: ModelTestCase,
        dataset_test_case: DatasetTestCase,
        fxt_dataset_root_dir: Path,
        fxt_tags: dict,
        fxt_num_repeat: int,
        tmpdir: pytest.TempdirFactory,
    ) -> None:
        for seed in range(fxt_num_repeat):
            test_case = RegressionTestCase(
                model=model_test_case,
                dataset=dataset_test_case,
                output_dir=Path(tmpdir) / str(seed),
            )

            run_name = f"{test_case.model.task}/{test_case.model.name}/{test_case.dataset.name}/{seed}"
            tags = {
                "task": test_case.model.task,
                "model": test_case.model.name,
                "dataset": test_case.dataset.name,
                "seed": str(seed),
                **fxt_tags,
            }
            data_root = (
                fxt_dataset_root_dir
                / test_case.model.task
                / test_case.dataset.data_root
            )
            with mlflow.start_run(tags=tags, run_name=run_name):
                overrides = [
                    f"+recipe={test_case.model.task}/{test_case.model.name}",
                    f"model.otx_model.config.roi_head.bbox_head.num_classes={test_case.dataset.num_classes}",
                    f"model.otx_model.config.roi_head.mask_head.num_classes={test_case.dataset.num_classes}",
                    f"data.data_root={data_root}",
                    f"data.data_format={test_case.dataset.data_format}",
                    f"base.output_dir={test_case.output_dir}",
                    f"seed={seed}",
                    "test=true",
                ] + [
                    f"{key}={value}"
                    for key, value in test_case.dataset.extra_overrides.items()
                ]
                if torch.cuda.is_available():
                    overrides.append("trainer=gpu")

                _, metrics = otx_train(overrides)

                # Submit metrics to MLFlow Tracker server
                mlflow.log_metrics(metrics)