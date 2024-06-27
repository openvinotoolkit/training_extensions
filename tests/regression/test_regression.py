# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.utils import run_main
import mlflow
import pandas as pd


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
                    "otx",
                    "train",
                    "--config",
                    f"src/otx/recipe/{test_case.model.task}/{test_case.model.name}.yaml",
                    "--model.num_classes",
                    str(test_case.dataset.num_classes),
                    "--data_root",
                    str(data_root),
                    "--data.data_format",
                    test_case.dataset.data_format,
                    "--work_dir",
                    str(test_case.output_dir),
                    "--engine.device",
                    fxt_accelerator,
                ]
                deterministic = test_case.dataset.extra_overrides.pop("deterministic", "False")
                for key, value in test_case.dataset.extra_overrides.items():
                    command_cfg.append(f"--{key}")
                    command_cfg.append(str(value))
                train_cfg = command_cfg.copy()
                train_cfg.extend(["--seed", str(seed)])
                train_cfg.extend(["--deterministic", deterministic])

                run_main(command_cfg=train_cfg, open_subprocess=True)
                checkpoint = test_case.output_dir / ".latest" / "train" / "best_checkpoint.ckpt"
                assert checkpoint.exists()

                test_cfg = command_cfg.copy()
                test_cfg[1] = "test"
                test_cfg += ["--checkpoint", str(checkpoint)]

                # TODO(harimkang): This command cannot create `metrics.csv`` file under test output directory
                # Without fixing this, we cannot submit the test metrics from the csv logged file
                run_main(command_cfg=test_cfg, open_subprocess=True)

                # This is also not working. It produces an empty dictionary for test_metrics = {}
                # with patch("sys.argv", test_cfg):
                #     cli = OTXCLI()
                #     test_metrics = cli.engine.trainer.callback_metrics
                # mlflow.log_metrics(test_metrics)

                # Submit metrics to MLFlow Tracker server
                for metric_csv_file in test_case.output_dir.glob("**/metrics.csv"):
                    self._submit_metric(metric_csv_file)

    def _submit_metric(self, metric_csv_file: Path) -> None:
        df = pd.read_csv(metric_csv_file)
        for step, sub_df in df.groupby("step"):
            sub_df = sub_df.drop("step", axis=1)

            for _, row in sub_df.iterrows():
                row = row.dropna()
                metrics = row.to_dict()
                mlflow.log_metrics(metrics=metrics, step=step)

        mlflow.log_artifact(local_path=str(metric_csv_file), artifact_path="metrics")


class TestMultiClassCls(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="classification/multi_class_cls", name="deit_tiny"),
        ModelTestCase(task="classification/multi_class_cls", name="dino_v2"),
        ModelTestCase(task="classification/multi_class_cls", name="efficientnet_b0"),
        ModelTestCase(task="classification/multi_class_cls", name="efficientnet_v2"),
        ModelTestCase(task="classification/multi_class_cls", name="mobilenet_v3_large"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name=f"multiclass_CUB_small_{idx}",
            data_root=Path("multiclass_classification/multiclass_CUB_small") / f"{idx}",
            data_format="imagenet_with_subset_dirs",
            num_classes=2,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.accuracy.MulticlassAccuracywithLabelGroup",
            },
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name=f"multiclass_CUB_medium",
            data_root=Path("multiclass_classification/multiclass_CUB_medium"),
            data_format="imagenet_with_subset_dirs",
            num_classes=67,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.accuracy.MulticlassAccuracywithLabelGroup",
            },
        ),
        DatasetTestCase(
            name=f"multiclass_food101_large",
            data_root=Path("multiclass_classification/multiclass_food101_large"),
            data_format="imagenet_with_subset_dirs",
            num_classes=20,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.accuracy.MulticlassAccuracywithLabelGroup",
            },
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
        ModelTestCase(task="classification/multi_label_cls", name="efficientnet_b0"),
        ModelTestCase(task="classification/multi_label_cls", name="efficientnet_v2"),
        ModelTestCase(task="classification/multi_label_cls", name="mobilenet_v3_large"),
        ModelTestCase(task="classification/multi_label_cls", name="deit_tiny"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name=f"multilabel_CUB_small_{idx}",
            data_root=Path("multilabel_classification/multilabel_CUB_small") / f"{idx}",
            data_format="datumaro",
            num_classes=3,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.accuracy.MultilabelAccuracywithLabelGroup",
            },
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name=f"multilabel_CUB_medium",
            data_root=Path("multilabel_classification/multilabel_CUB_medium"),
            data_format="datumaro",
            num_classes=68,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.accuracy.MultilabelAccuracywithLabelGroup",
            },
        ),
        DatasetTestCase(
            name=f"multilabel_food101_large",
            data_root=Path("multilabel_classification/multilabel_food101_large"),
            data_format="datumaro",
            num_classes=21,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.accuracy.MultilabelAccuracywithLabelGroup",
            },
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
        ModelTestCase(task="classification/h_label_cls", name="efficientnet_b0"),
        ModelTestCase(task="classification/h_label_cls", name="efficientnet_v2"),
        ModelTestCase(task="classification/h_label_cls", name="mobilenet_v3_large"),
        ModelTestCase(task="classification/h_label_cls", name="deit_tiny"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name=f"hlabel_CUB_small_{idx}",
            data_root=Path("hlabel_classification/hlabel_CUB_small") / f"{idx}",
            data_format="datumaro",
            num_classes=6,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.accuracy.HlabelAccuracy",
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
                "deterministic": "True",
                "metric": "otx.core.metrics.accuracy.HlabelAccuracy",
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
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name="pothole_medium",
            data_root=Path("detection/pothole_medium"),
            data_format="coco",
            num_classes=1,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
        ),
        DatasetTestCase(
            name="vitens_large",
            data_root=Path("detection/vitens_large"),
            data_format="coco",
            num_classes=1,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
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
            extra_overrides={},
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name="kvasir_medium",
            data_root=Path("semantic_seg/kvasir_medium"),
            data_format="common_semantic_segmentation_with_subset_dirs",
            num_classes=2,
            extra_overrides={},
        ),
        DatasetTestCase(
            name="kvasir_large",
            data_root=Path("semantic_seg/kvasir_large"),
            data_format="common_semantic_segmentation_with_subset_dirs",
            num_classes=2,
            extra_overrides={},
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
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name="coco_car_person_medium",
            data_root=Path("instance_seg/coco_car_person_medium"),
            data_format="coco",
            num_classes=2,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
        ),
        DatasetTestCase(
            name="vitens_coliform",
            data_root=Path("instance_seg/Vitens-Coliform-coco"),
            data_format="coco",
            num_classes=1,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
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
        ModelTestCase(task="visual_prompting", name="sam_vit_b"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [  # noqa: RUF012
        DatasetTestCase(
            name=f"wgisd_small_{idx}",
            data_root=Path("visual_prompting/wgisd_small") / f"{idx}",
            data_format="coco",
            num_classes=5,
            extra_overrides={"deterministic": "warn"},
        )
        for idx in range(1, 4)
    ] + [
        DatasetTestCase(
            name="coco_car_person_medium",
            data_root=Path("visual_prompting/coco_car_person_medium"),
            data_format="coco",
            num_classes=2,
            extra_overrides={"deterministic": "warn"},
        ),
        DatasetTestCase(
            name="vitens_coliform",
            data_root=Path("visual_prompting/Vitens-Coliform-coco"),
            data_format="coco",
            num_classes=1,
            extra_overrides={"deterministic": "warn"},
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


class TestZeroShotVisualPrompting(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="zero_shot_visual_prompting", name="sam_tiny_vit"),
        ModelTestCase(task="zero_shot_visual_prompting", name="sam_vit_b"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [
        DatasetTestCase(
            name="coco_car_person_medium_datumaro",
            data_root=Path("zero_shot_visual_prompting/coco_car_person_medium_datumaro"),
            data_format="datumaro",
            num_classes=2,
            extra_overrides={"max_epochs": "1", "deterministic": "warn"},
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


class TestTileObjectDetection(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="detection", name="atss_mobilenetv2_tile"),
        ModelTestCase(task="detection", name="ssd_mobilenetv2_tile"),
        ModelTestCase(task="detection", name="yolox_tiny_tile"),
        ModelTestCase(task="detection", name="yolox_s_tile"),
        ModelTestCase(task="detection", name="yolox_l_tile"),
        ModelTestCase(task="detection", name="yolox_x_tile"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [
        DatasetTestCase(
            name="vitens_coliform",
            data_root=Path("instance_seg/Vitens-Coliform-coco"),
            data_format="coco",
            num_classes=1,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
        ),
        DatasetTestCase(
            name="vitens_aeromonas",
            data_root=Path("instance_seg/Vitens-Aeromonas-coco"),
            data_format="coco",
            num_classes=1,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
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


class TestTileInstanceSegmentation(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="instance_segmentation", name="maskrcnn_efficientnetb2b_tile"),
        ModelTestCase(task="instance_segmentation", name="maskrcnn_r50_tile"),
        ModelTestCase(task="instance_segmentation", name="maskrcnn_swint_tile"),
    ]
    # Test case parametrization for dataset
    DATASET_TEST_CASES = [
        DatasetTestCase(
            name="vitens_coliform",
            data_root=Path("instance_seg/Vitens-Coliform-coco"),
            data_format="coco",
            num_classes=1,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
        ),
        DatasetTestCase(
            name="vitens_aeromonas",
            data_root=Path("instance_seg/Vitens-Aeromonas-coco"),
            data_format="coco",
            num_classes=1,
            extra_overrides={
                "deterministic": "True",
                "metric": "otx.core.metrics.fmeasure.FMeasureCallable",
                "callback_monitor": "val/f1-score",
                "scheduler.monitor": "val/f1-score",
            },
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


class TestActionClassification(BaseTest):
    # Test case parametrization for model
    MODEL_TEST_CASES = [  # noqa: RUF012
        ModelTestCase(task="action_classification", name="x3d"),
        ModelTestCase(task="action_classification", name="movinet"),
    ]
    DATASET_TEST_CASES = [
        DatasetTestCase(
            name="ucf-5percent",
            data_root=Path("action_classification/ucf-kinetics-5percent"),
            data_format="kinetics",
            num_classes=101,
            extra_overrides={"max_epochs": "10", "deterministic": "True"},
        ),
        DatasetTestCase(
            name="ucf-30percent",
            data_root=Path("action_classification/ucf-kinetics-30percent"),
            data_format="kinetics",
            num_classes=101,
            extra_overrides={"max_epochs": "10", "deterministic": "True"},
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
