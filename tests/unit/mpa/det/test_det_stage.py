import os

import pytest

from otx.mpa import Stage
from otx.mpa.det.stage import DetectionStage
from otx.mpa.utils.config_utils import MPAConfig
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_RECIPE_CONFIG_PATH,
    DEFAULT_DET_TEMPLATE_DIR,
)


class TestOTXDetectionStage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(DEFAULT_DET_RECIPE_CONFIG_PATH)
        self.stage = DetectionStage(name="", mode="train", config=cfg, common_cfg=None, index=0)
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_configure(self, mocker):
        mock_cfg_model = mocker.patch.object(DetectionStage, "configure_model")
        mock_cfg_ckpt = mocker.patch.object(DetectionStage, "configure_ckpt")
        # mock_cfg_data = mocker.patch.object(DetectionStage, "configure_data")
        mock_cfg_regularization = mocker.patch.object(DetectionStage, "configure_regularization")
        # mock_cfg_hyperparams = mocker.patch.object(DetectionStage, "configure_hyperparams")
        mock_cfg_task = mocker.patch.object(DetectionStage, "configure_task")
        mock_cfg_hook = mocker.patch.object(DetectionStage, "configure_hook")

        fake_arg = {"pretrained": True, "foo": "bar"}
        returned_value = self.stage.configure(self.model_cfg, "", self.data_cfg, True, **fake_arg)
        mock_cfg_model.assert_called_once_with(self.stage.cfg, self.model_cfg, True, **fake_arg)
        mock_cfg_ckpt.assert_called_once_with(self.stage.cfg, "", fake_arg.get("pretrained", None))
        # mock_cfg_data.assert_called_once_with(self.stage.cfg, True, self.data_cfg)
        mock_cfg_regularization.assert_called_once_with(self.stage.cfg, True)
        # mock_cfg_hyperparams.assert_called_once_with(self.stage.cfg, True)
        mock_cfg_task.assert_called_once_with(self.stage.cfg, True, **fake_arg)
        mock_cfg_hook.assert_called_once_with(self.stage.cfg)

        assert returned_value == self.stage.cfg

    @e2e_pytest_unit
    def test_configure_model(self):
        fake_arg = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        self.stage.configure_model(self.stage.cfg, self.model_cfg, True, **fake_arg)

        assert self.stage.cfg.model_task

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        mock_super_cfg_data = mocker.patch.object(Stage, "configure_data")
        self.stage.configure_data(self.stage.cfg, True, self.data_cfg, pretrained=None)

        mock_super_cfg_data.assert_called_once()
        assert self.stage.cfg.data
        assert self.stage.cfg.data.train
        assert self.stage.cfg.data.val

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        self.stage.data_classes = ["red", "green"]
        self.stage.model_classes = []
        mock_cfg_classes = mocker.patch.object(DetectionStage, "configure_classes")
        mocker.patch.object(DetectionStage, "configure_bbox_head")
        self.stage.configure_task(self.stage.cfg, True)

        mock_cfg_classes.assert_called_once()
