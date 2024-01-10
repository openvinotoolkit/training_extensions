# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


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
        fxt_accelerator: str,
        tmpdir: pytest.TempdirFactory,
        head_name: str,
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
                    f"model.otx_model.config.{head_name}.num_classes={test_case.dataset.num_classes}",
                    f"data.data_root={data_root}",
                    f"data.data_format={test_case.dataset.data_format}",
                    f"base.output_dir={test_case.output_dir}",
                    f"seed={seed}",
                    f"trainer={fxt_accelerator}",
                    "test=true",
                    "trainer=gpu",
                ] + [
                    f"{key}={value}"
                    for key, value in test_case.dataset.extra_overrides.items()
                ]
                metrics = otx_train(overrides)

                # Submit metrics to MLFlow Tracker server
                mlflow.log_metrics(metrics)


class TestMultiClassCls(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="multiclass_classification", name="otx_deit_tiny"),
        ModelTestCase(task="multiclass_classification", name="otx_dino_v2"),
        ModelTestCase(task="multiclass_classification", name="otx_efficientnet_b0"),
        ModelTestCase(task="multiclass_classification", name="otx_efficientnet_v2"),
        ModelTestCase(task="multiclass_classification", name="otx_mobilenet_v3_large"),
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
    DATASET_TEST_CASES.extend([
        DatasetTestCase(
            name=f"multiclass_CUB_medium",
            data_root=Path("multiclass_CUB_medium"),
            data_format="imagenet_with_subset_dirs",
            num_classes=67,
            extra_overrides={"trainer.max_epochs": "20"},
        ),
        DatasetTestCase(
            name=f"multiclass_food101_large",
            data_root=Path("multiclass_food101_large"),
            data_format="imagenet_with_subset_dirs",
            num_classes=20,
            extra_overrides={"trainer.max_epochs": "20"},
        )
    ])

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
        fxt_accelerator: str,
        tmpdir: pytest.TempdirFactory,
    ) -> None:
        self._test_regression(
            model_test_case=model_test_case,
            dataset_test_case=dataset_test_case,
            fxt_dataset_root_dir=fxt_dataset_root_dir,
            fxt_tags=fxt_tags,
            fxt_num_repeat=fxt_num_repeat,
            fxt_accelerator=fxt_accelerator,
            tmpdir=tmpdir,
            head_name="head",
        )


class TestMultilabelCls(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="multilabel_classification", name="efficientnet_b0_light"),
        ModelTestCase(task="multilabel_classification", name="efficientnet_v2_light"),
        ModelTestCase(task="multilabel_classification", name="mobilenet_v3_large_light"),
        ModelTestCase(task="multilabel_classification", name="otx_deit_tiny"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name=f"multilabel_CUB_small_{idx}",
            data_root=Path("multilabel_CUB_small") / f"{idx}",
            data_format="datumaro",
            num_classes=3,
            extra_overrides={"trainer.max_epochs": "20"},
        )
        for idx in range(1, 4)
    ]
    DATASET_TEST_CASES.extend([
        DatasetTestCase(
            name=f"multilabel_CUB_medium",
            data_root=Path("multilabel_CUB_medium"),
            data_format="datumaro",
            num_classes=68,
            extra_overrides={"trainer.max_epochs": "20"},
        ),
        DatasetTestCase(
            name=f"multilabel_food101_large",
            data_root=Path("multilabel_food101_large"),
            data_format="datumaro",
            num_classes=21,
            extra_overrides={"trainer.max_epochs": "20"},
        )
    ])

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
        fxt_accelerator: str,
        tmpdir: pytest.TempdirFactory,
    ) -> None:
        self._test_regression(
            model_test_case=model_test_case,
            dataset_test_case=dataset_test_case,
            fxt_dataset_root_dir=fxt_dataset_root_dir,
            fxt_tags=fxt_tags,
            fxt_num_repeat=fxt_num_repeat,
            fxt_accelerator=fxt_accelerator,
            tmpdir=tmpdir,
            head_name="head",
        )


class TestObjectDetection(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="detection", name="atss_mobilenetv2"),
        ModelTestCase(task="detection", name="atss_resnext101"),
        ModelTestCase(task="detection", name="ssd_mobilenetv2"),
        ModelTestCase(task="detection", name="yolox_tiny"),
        ModelTestCase(task="detection", name="yolox_s"),
        ModelTestCase(task="detection", name="yolox_l"),
        ModelTestCase(task="detection", name="yolox_x"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name=f"pothole_small_{idx}",
            data_root=Path("pothole_small") / f"{idx}",
            data_format="coco",
            num_classes=1,
            extra_overrides={"trainer.max_epochs": "10"},
        )
        for idx in range(1, 4)
    ]
    DATASET_TEST_CASES.extend(
        [
            DatasetTestCase(
                name="pothole_medium",
                data_root="pothole_medium",
                data_format="coco",
                num_classes=1,
                extra_overrides={"trainer.max_epochs": "10"}
            ),
            DatasetTestCase(
                name="vitens_large",
                data_root="vitens_large",
                data_format="coco",
                num_classes=1,
                extra_overrides={"trainer.max_epochs": "10"}
            ),
        ]
    )

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
            head_name="bbox_head",
        )
