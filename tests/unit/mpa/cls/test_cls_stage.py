# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest

from otx.mpa import Stage
from otx.mpa.cls.stage import ClsStage
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import setup_mpa_task_parameters


class TestOTXClsStage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.model_cfg, self.data_cfg, recipie_cfg = setup_mpa_task_parameters(
            task_type="incremental", create_test=True, create_val=True
        )
        self.stage = ClsStage(name="", mode="train", config=recipie_cfg, common_cfg=None, index=0)

    @e2e_pytest_unit
    def test_configure(self, mocker):
        mock_cfg_model = mocker.patch.object(ClsStage, "configure_model")
        mock_cfg_ckpt = mocker.patch.object(ClsStage, "configure_ckpt")
        mock_cfg_data = mocker.patch.object(ClsStage, "configure_data")
        mock_cfg_task = mocker.patch.object(ClsStage, "configure_task")

        fake_arg = {"pretrained": True, "foo": "bar"}
        returned_value = self.stage.configure(self.model_cfg, "", self.data_cfg, True, **fake_arg)
        mock_cfg_model.assert_called_once_with(self.stage.cfg, self.model_cfg, True, **fake_arg)
        mock_cfg_ckpt.assert_called_once_with(self.stage.cfg, "", fake_arg.get("pretrained", None))
        mock_cfg_data.assert_called_once_with(self.stage.cfg, self.data_cfg, True, **fake_arg)
        mock_cfg_task.assert_called_once_with(self.stage.cfg, True, **fake_arg)

        assert returned_value == self.stage.cfg

    @e2e_pytest_unit
    def test_configure_model(self):
        fake_arg = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        self.stage.configure_model(self.stage.cfg, self.model_cfg, True, **fake_arg)

        assert self.stage.cfg.model_task

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        mock_super_cfg_data = mocker.patch.object(Stage, "configure_data")
        self.stage.configure_data(self.stage.cfg, self.data_cfg, True, pretrained=None)

        mock_super_cfg_data.assert_called_once()
        assert self.stage.cfg.data
        assert self.stage.cfg.data.train
        assert self.stage.cfg.data.val

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        mock_cfg_classes = mocker.patch.object(ClsStage, "configure_classes")
        self.stage.configure_task(self.stage.cfg, True)

        mock_cfg_classes.assert_called_once()

    @e2e_pytest_unit
    def test_configure_classes(self, mocker):
        mocker.patch.object(Stage, "get_model_classes", return_value=["foo", "bar"])
        mocker.patch.object(Stage, "get_data_cfg", return_value=self.data_cfg)
        self.stage.configure_classes(self.stage.cfg)

        assert self.stage.model_classes == ["foo", "bar"]

    @e2e_pytest_unit
    def test_configure_configure_topk(self):
        self.stage.cfg.model.head.num_classes = 2
        self.stage.configure_topk(self.stage.cfg)

        assert self.stage.cfg.model.head.topk == (1,)
