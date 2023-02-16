# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import tempfile

from mmcv.runner import EpochBasedRunner
from mmcv.utils import get_logger
from torch.optim import SGD

from otx.algorithms.common.adapters.mmcv.utils import build_data_parallel
from otx.algorithms.segmentation.adapters.mmseg.nncf.hooks import (
    CustomstepLrUpdaterHook,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.common.adapters.mmcv.nncf.test_helpers import (
    create_config,
    create_model,
)


class TestCustomstepLrUpdaterHook:
    @e2e_pytest_unit
    def test_get_regular_lr(self):
        mock_config = create_config()
        mock_model = create_model()

        with tempfile.TemporaryDirectory() as tempdir:
            runner = EpochBasedRunner(
                build_data_parallel(mock_model, mock_config),
                optimizer=SGD(mock_model.parameters(), lr=0.01, momentum=0.9),
                max_epochs=10,
                work_dir=tempdir,
                logger=get_logger("mmcv"),
            )
            runner.data_loader = range(10)

            hook = CustomstepLrUpdaterHook(1)
            hook.base_lr = [0.01, 0.1]
            hook.before_train_iter(runner)
            assert hook.get_regular_lr(runner) == [0.01, 0.1]
            runner._iter = 20
            assert hook.get_regular_lr(runner) == [0.01 * hook.gamma, 0.1 * hook.gamma]

            runner = EpochBasedRunner(
                build_data_parallel(mock_model, mock_config),
                optimizer={
                    "dummy": SGD(mock_model.parameters(), lr=0.01, momentum=0.9),
                },
                max_epochs=10,
                work_dir=tempdir,
                logger=get_logger("mmcv"),
            )
            runner.data_loader = range(10)

            hook = CustomstepLrUpdaterHook(1)
            hook.base_lr = {"dummy": [0.01, 0.1]}
            hook.before_train_iter(runner)
            assert hook.get_regular_lr(runner) == {"dummy": [0.01, 0.1]}
            runner._iter = 20
            assert hook.get_regular_lr(runner) == {"dummy": [0.01 * hook.gamma, 0.1 * hook.gamma]}

    @e2e_pytest_unit
    def test_get_fixed_lr(self):
        mock_config = create_config()
        mock_model = create_model()

        with tempfile.TemporaryDirectory() as tempdir:
            runner = EpochBasedRunner(
                build_data_parallel(mock_model, mock_config),
                optimizer=SGD(mock_model.parameters(), lr=0.01, momentum=0.9),
                max_epochs=10,
                work_dir=tempdir,
                logger=get_logger("mmcv"),
            )
            runner.data_loader = range(10)

            hook = CustomstepLrUpdaterHook(1, fixed="constant", fixed_iters=5, fixed_ratio=0.2)
            hook.base_lr = [1.0]
            hook.before_train_iter(runner)
            lrs = []
            for i in range(1, 25):
                lrs.extend(hook.get_fixed_lr(i, [1.0]))
            assert len(set(lrs)) == 1

            hook = CustomstepLrUpdaterHook(1, fixed="linear", fixed_iters=5, fixed_ratio=0.2)
            hook.base_lr = [1.0]
            hook.before_train_iter(runner)
            lrs = []
            for i in range(1, 25):
                lrs.extend(hook.get_fixed_lr(i, [1.0]))
            assert sorted(lrs) == lrs

            hook = CustomstepLrUpdaterHook(1, fixed="cos", fixed_iters=5, fixed_ratio=0.2)
            hook.base_lr = [1.0]
            hook.before_train_iter(runner)
            lrs = []
            for i in range(1, 25):
                lrs.extend(hook.get_fixed_lr(i, [1.0]))
            assert sorted(lrs) == lrs

    @e2e_pytest_unit
    def test_get_wramup_lr(self):
        mock_config = create_config()
        mock_model = create_model()

        with tempfile.TemporaryDirectory() as tempdir:
            runner = EpochBasedRunner(
                build_data_parallel(mock_model, mock_config),
                optimizer=SGD(mock_model.parameters(), lr=0.01, momentum=0.9),
                max_epochs=10,
                work_dir=tempdir,
                logger=get_logger("mmcv"),
            )
            runner.data_loader = range(10)

            hook = CustomstepLrUpdaterHook(1, warmup="constant", warmup_iters=5, warmup_ratio=0.2)
            hook.base_lr = [1.0]
            hook.before_train_iter(runner)
            lrs = []
            for i in range(1, 25):
                lrs.extend(hook.get_warmup_lr(i, [1.0]))
            assert len(set(lrs)) == 1

            hook = CustomstepLrUpdaterHook(1, warmup="linear", warmup_iters=5, warmup_ratio=0.2)
            hook.base_lr = [1.0]
            hook.before_train_iter(runner)
            lrs = []
            for i in range(1, 25):
                lrs.extend(hook.get_warmup_lr(i, [1.0]))
            assert sorted(lrs) == lrs
