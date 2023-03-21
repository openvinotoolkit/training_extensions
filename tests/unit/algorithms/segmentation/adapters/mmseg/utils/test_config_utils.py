# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv import Config

from otx.algorithms.segmentation.adapters.mmseg.utils.config_utils import (
    patch_adaptive_repeat_dataset,
    patch_config,
    patch_datasets,
    patch_evaluation,
    patch_model_config,
    set_hyperparams,
)
from otx.algorithms.segmentation.configs.base import SegmentationConfig
from otx.api.entities.label import Domain, LabelEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def _create_dummy_config() -> Config:

    config: dict = dict(
        model=dict(
            type="ClassIncrEncoderDecoder",
            backbone=dict(),
            decode_head=dict(
                norm_cfg=dict(type="BN", requires_grad=True),
                loss_decode=[dict(type="CrossEntropyLoss")],
            ),
        ),
        data=dict(
            train=dict(type="RepeatDataset", times=1, adaptive_repeat=True, dataset=dict(pipeline=[])),
            val=dict(type="", pipeline=[]),
            test=dict(type="", pipeline=[]),
        ),
        runner=dict(type="EpochRunnerWithCancel", max_epochs=300),
        checkpoint_config=dict(interval=1),
        optimizer=dict(),
        lr_config=dict(policy="ReduceLROnPlateau", warmup_iters=80, fixed_iters=100),
        evaluation=dict(interval=1, metric=["mIoU", "mDice"]),
        custom_hooks=[],
        params_config=dict(iters=0),
    )

    return Config(config)


class TestConfigUtilsValidation:
    @e2e_pytest_unit
    def test_patch_config(self) -> None:
        config: Config = _create_dummy_config()
        labels: list[LabelEntity] = [LabelEntity(name="test label", domain=Domain.SEGMENTATION)]
        num_classes: int = len(labels) + 1
        work_dir: str = "./work_dir"
        patch_config(config, work_dir, labels)

        assert "custom_hooks" in config
        assert "train_pipeline" not in config
        assert config.work_dir == work_dir
        assert len(config.data["train"].classes) == num_classes

    @e2e_pytest_unit
    def test_patch_model_config_nondistributed(self) -> None:
        config: Config = _create_dummy_config()
        labels: list[LabelEntity] = [LabelEntity(name="test label", domain=Domain.SEGMENTATION)]
        num_classes: int = len(labels) + 1
        distributed: bool = False
        patch_model_config(config, labels, distributed)

        assert config.model.decode_head["num_classes"] == num_classes
        assert config.model.decode_head.norm_cfg.type != "SyncBN"

    @e2e_pytest_unit
    def test_patch_datasets(self) -> None:
        config: Config = _create_dummy_config()
        patch_datasets(config, type="MPASegDataset")

        assert "classes" not in config.data.train.dataset
        assert "labels" in config.data.train.dataset

    @e2e_pytest_unit
    def test_patch_evaluation(self) -> None:
        config: Config = _create_dummy_config()
        patch_evaluation(config)

        assert config.early_stop_metric == "mDice"

    @e2e_pytest_unit
    def test_set_hyperparams(self) -> None:
        config: Config = _create_dummy_config()
        hyperparameters: SegmentationConfig = SegmentationConfig()
        set_hyperparams(config, hyperparameters)

        assert config.runner.max_epochs > 0

    @e2e_pytest_unit
    def test_patch_adaptive_repeat_dataset(self) -> None:
        config: Config = _create_dummy_config()
        num_samples: int = 10

        initial_epochs: int = config.runner.max_epochs
        patch_adaptive_repeat_dataset(config, num_samples)
        epochs_after_adaptive_repeat: int = config.runner.max_epochs

        assert initial_epochs > epochs_after_adaptive_repeat
