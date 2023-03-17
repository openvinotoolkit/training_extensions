# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest

from otx.mpa.cls.semisl.stage import SemiSLClsStage
from otx.mpa.cls.stage import ClsStage
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import setup_mpa_task_parameters


class TestOTXSemiSLClsStage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.model_cfg, self.data_cfg, recipie_cfg = setup_mpa_task_parameters(
            task_type="semisl", create_val=True, create_test=True
        )
        self.stage = SemiSLClsStage(name="", mode="train", config=recipie_cfg, common_cfg=None, index=0)

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        mock_ul_dataloader = mocker.patch.object(ClsStage, "configure_unlabeled_dataloader")
        fake_semisl_data_cfg = {"data": {"unlabeled": {"otx_dataset": "foo"}}}
        self.stage.configure_data(self.stage.cfg, fake_semisl_data_cfg, True)

        mock_ul_dataloader.assert_called_once()

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        self.stage.cfg.merge_from_dict(self.model_cfg)
        mock_cfg_classes = mocker.patch.object(ClsStage, "configure_classes")
        self.stage.configure_task(self.stage.cfg, True)

        assert "task_adapt" not in self.stage.cfg.model
        mock_cfg_classes.assert_called_once()
