# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from otx.cli.cli import OTXCLI
from unittest.mock import patch

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
                # / test_case.model.task
                / test_case.dataset.data_root
            )
            with mlflow.start_run(tags=tags, run_name=run_name):
                command_cfg = [
                    "otx", "train",
                    "--config", f"src/otx/recipe/{test_case.model.task}/{test_case.model.name}.yaml",
                    "--model.num_classes", str(test_case.dataset.num_classes),
                    "--data_root", str(data_root),
                    "--data.config.data_format", test_case.dataset.data_format,
                    "--engine.work_dir", str(test_case.output_dir),
                    "--engine.device", fxt_accelerator,
                ]
                deterministic = test_case.dataset.extra_overrides.pop("deterministic", "False")
                for key, value in test_case.dataset.extra_overrides.items():
                    command_cfg.append(f"--{key}")
                    command_cfg.append(str(value))
                train_cfg = command_cfg.copy()
                train_cfg.extend(["--seed", str(seed)])
                train_cfg.extend(["--deterministic", deterministic])
                with patch("sys.argv", train_cfg):
                    cli = OTXCLI()
                    train_metrics = cli.engine.trainer.callback_metrics
                    checkpoint = cli.engine.checkpoint
                command_cfg[1] = "test"
                command_cfg += ["--checkpoint", checkpoint]
                with patch("sys.argv", command_cfg):
                    cli = OTXCLI()
                    test_metrics = cli.engine.trainer.callback_metrics
                metrics = {**train_metrics, **test_metrics}

                # Submit metrics to MLFlow Tracker server
                mlflow.log_metrics(metrics)


class TestMultiClassCls(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="classification/multi_class_cls", name="otx_deit_tiny"),
        ModelTestCase(task="classification/multi_class_cls", name="otx_dino_v2"),
        ModelTestCase(task="classification/multi_class_cls", name="otx_efficientnet_b0"),
        ModelTestCase(task="classification/multi_class_cls", name="otx_efficientnet_v2"),
        ModelTestCase(task="classification/multi_class_cls", name="otx_mobilenet_v3_large"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name=f"multiclass_CUB_small_{idx}",
            data_root=Path("multiclass_classification/multiclass_CUB_small") / f"{idx}",
            data_format="imagenet_with_subset_dirs",
            num_classes=2,
            extra_overrides={"max_epochs": "20"},
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name=f"multiclass_CUB_medium",
            data_root=Path("multiclass_classification/multiclass_CUB_medium"),
            data_format="imagenet_with_subset_dirs",
            num_classes=67,
            extra_overrides={"max_epochs": "20"},
        ),
        DatasetTestCase(
            name=f"multiclass_food101_large",
            data_root=Path("multiclass_classification/multiclass_food101_large"),
            data_format="imagenet_with_subset_dirs",
            num_classes=20,
            extra_overrides={"max_epochs": "20"},
        )
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
        )


class TestMultilabelCls(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="classification/multi_label_cls", name="efficientnet_b0_light"),
        ModelTestCase(task="classification/multi_label_cls", name="efficientnet_v2_light"),
        ModelTestCase(task="classification/multi_label_cls", name="mobilenet_v3_large_light"),
        ModelTestCase(task="classification/multi_label_cls", name="otx_deit_tiny"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name=f"multilabel_CUB_small_{idx}",
            data_root=Path("multilabel_classification/multilabel_CUB_small") / f"{idx}",
            data_format="datumaro",
            num_classes=3,
            extra_overrides={"max_epochs": "20"},
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name=f"multilabel_CUB_medium",
            data_root=Path("multilabel_classification/multilabel_CUB_medium"),
            data_format="datumaro",
            num_classes=68,
            extra_overrides={"max_epochs": "20"},
        ),
        DatasetTestCase(
            name=f"multilabel_food101_large",
            data_root=Path("multilabel_classification/multilabel_food101_large"),
            data_format="datumaro",
            num_classes=21,
            extra_overrides={"max_epochs": "20"},
        )
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
        )


class TestHlabelCls(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="classification/h_label_cls", name="efficientnet_b0_light"),
        ModelTestCase(task="classification/h_label_cls", name="efficientnet_v2_light"),
        ModelTestCase(task="classification/h_label_cls", name="mobilenet_v3_large_light"),
        ModelTestCase(task="classification/h_label_cls", name="otx_deit_tiny"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name=f"hlabel_CUB_small_{idx}",
            data_root=Path("hlabel_classification/hlabel_CUB_small") / f"{idx}",
            data_format="datumaro",
            num_classes=6,
            extra_overrides={
                "max_epochs": "20",
                "model.num_multiclass_heads": "3",
                "model.num_multilabel_classes": "0",
            },
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name=f"hlabel_CUB_medium",
            data_root=Path("hlabel_classification/hlabel_CUB_medium"),
            data_format="datumaro",
            num_classes=102,
            extra_overrides={
                "max_epochs": "20",
                "model.num_multiclass_heads": "23",
                "model.num_multilabel_classes": "0",
            },
        )
        
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
            data_root=Path("detection/pothole_small") / f"{idx}",
            data_format="coco",
            num_classes=1,
            extra_overrides={"max_epochs": "40", "deterministic": "True"},
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name="pothole_medium",
            data_root=Path("detection/pothole_medium"),
            data_format="coco",
            num_classes=1,
            extra_overrides={"max_epochs": "40", "deterministic": "True"}
        ),
        DatasetTestCase(
            name="vitens_large",
            data_root=Path("detection/vitens_large"),
            data_format="coco",
            num_classes=1,
            extra_overrides={"max_epochs": "40", "deterministic": "True"}
        )
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
        )

class TestSemanticSegmentation(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="semantic_segmentation", name="litehrnet_18"),
        ModelTestCase(task="semantic_segmentation", name="litehrnet_s"),
        ModelTestCase(task="semantic_segmentation", name="litehrnet_x"),
        ModelTestCase(task="semantic_segmentation", name="segnext_b"),
        ModelTestCase(task="semantic_segmentation", name="segnext_s"),
        ModelTestCase(task="semantic_segmentation", name="segnext_t"),
        ModelTestCase(task="semantic_segmentation", name="dino_v2"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name=f"kvasir_small_{idx}",
            data_root=Path("semantic_seg/kvasir_small") / f"{idx}",
            data_format="common_semantic_segmentation_with_subset_dirs",
            num_classes=2,
            extra_overrides={"max_epochs": "40"},
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name="kvasir_medium",
            data_root=Path("semantic_seg/kvasir_medium"),
            data_format="common_semantic_segmentation_with_subset_dirs",
            num_classes=2,
            extra_overrides={"max_epochs": "40"}
        ),
        DatasetTestCase(
            name="kvasir_large",
            data_root=Path("semantic_seg/kvasir_large"),
            data_format="common_semantic_segmentation_with_subset_dirs",
            num_classes=2,
            extra_overrides={"max_epochs": "40"}
        )
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
            name=f"wgisd_small_{idx}",
            data_root=Path("instance_seg/wgisd_small") / f"{idx}",
            data_format="coco",
            num_classes=5,
            extra_overrides={"max_epochs": "20", "deterministic": "True"},
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name="coco_car_person_medium",
            data_root=Path("instance_seg/coco_car_person_medium"),
            data_format="coco",
            num_classes=2,
            extra_overrides={"max_epochs": "20", "deterministic": "True"}
        ),
        DatasetTestCase(
            name="vitens_coliform",
            data_root=Path("instance_seg/Vitens-Coliform-coco"),
            data_format="coco",
            num_classes=1,
            extra_overrides={"max_epochs": "20", "deterministic": "True"}
        )
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
        )


class TestVisualPrompting(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="visual_prompting", name="sam_tiny_vit"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name=f"wgisd_small_{idx}",
            data_root=Path("visual_prompting/wgisd_small") / f"{idx}",
            data_format="coco",
            num_classes=5,
            extra_overrides={"max_epochs": "20", "deterministic": "True"},
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name="coco_car_person_medium",
            data_root=Path("visual_prompting/coco_car_person_medium"),
            data_format="coco",
            num_classes=2,
            extra_overrides={"max_epochs": "20", "deterministic": "True"}
        ),
        DatasetTestCase(
            name="vitens_coliform",
            data_root=Path("visual_prompting/Vitens-Coliform-coco"),
            data_format="coco",
            num_classes=1,
            extra_overrides={"max_epochs": "20", "deterministic": "True"}
        )
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
        )
