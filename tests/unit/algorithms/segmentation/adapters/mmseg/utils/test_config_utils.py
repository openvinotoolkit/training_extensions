# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv import Config

from otx.algorithms.segmentation.adapters.mmseg.utils.config_utils import (
    patch_datasets,
    patch_evaluation,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def _create_dummy_config() -> Config:

    config: dict = dict(
        model=dict(
            type="OTXEncoderDecoder",
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
