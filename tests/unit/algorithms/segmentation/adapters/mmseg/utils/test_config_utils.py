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
    def test_patch_datasets(self) -> None:
        config: Config = _create_dummy_config()
        patch_datasets(config, type="MPASegDataset")

        assert "classes" not in config.data.train.dataset
        assert "labels" in config.data.train.dataset

    @e2e_pytest_unit
    def test_patch_datasets_disable_memcache_for_test_subset(self) -> None:
        """Test patch_datasets function to check memcache disabled for test set."""
        config = Config(
            dict(
                data=dict(
                    train=dict(pipeline=[dict(type="LoadImageFromFile")]),
                    val=dict(pipeline=[dict(type="LoadImageFromFile")]),
                    test=dict(pipeline=[dict(type="LoadImageFromFile")]),
                    unlabeled=dict(pipeline=[dict(type="LoadImageFromFile")]),
                )
            )
        )
        patch_datasets(config, type="FakeType")
        assert config.data.train.pipeline[0].type == "LoadImageFromOTXDataset"
        assert config.data.train.pipeline[0].enable_memcache == True
        assert config.data.val.pipeline[0].type == "LoadImageFromOTXDataset"
        assert config.data.val.pipeline[0].enable_memcache == True
        assert config.data.test.pipeline[0].type == "LoadImageFromOTXDataset"
        assert getattr(config.data.test.pipeline[0], "enable_memcache", False) == False
        # Note: cannot set enable_memcache attr due to mmdeploy error
        assert config.data.unlabeled.pipeline[0].type == "LoadImageFromOTXDataset"
        assert config.data.unlabeled.pipeline[0].enable_memcache == True

    @e2e_pytest_unit
    def test_patch_evaluation(self) -> None:
        config: Config = _create_dummy_config()
        patch_evaluation(config)

        assert config.early_stop_metric == "mDice"
