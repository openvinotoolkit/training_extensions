import os

import pytest

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.mpa import Stage
from otx.mpa.seg.stage import SegStage
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.segmentation.test_helpers import (
    DEFAULT_RECIPE_CONFIG_PATH,
    DEFAULT_SEG_TEMPLATE_DIR,
)


class TestOTXSegStage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(DEFAULT_RECIPE_CONFIG_PATH)
        self.stage = SegStage(name="", mode="train", config=cfg, common_cfg=None, index=0)
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_configure(self, mocker):
        mock_cfg_model = mocker.patch.object(SegStage, "configure_model")
        mock_cfg_ckpt = mocker.patch.object(SegStage, "configure_ckpt")
        mock_cfg_data = mocker.patch.object(SegStage, "configure_data")
        mock_cfg_task = mocker.patch.object(SegStage, "configure_task")
        mock_cfg_hook = mocker.patch.object(SegStage, "configure_hook")

        fake_arg = {"pretrained": True, "foo": "bar"}
        returned_value = self.stage.configure(self.model_cfg, "", self.data_cfg, True, **fake_arg)
        mock_cfg_model.assert_called_once_with(self.stage.cfg, self.model_cfg, True, **fake_arg)
        mock_cfg_ckpt.assert_called_once_with(self.stage.cfg, "", fake_arg.get("pretrained", None))
        mock_cfg_data.assert_called_once_with(self.stage.cfg, True, self.data_cfg)
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
        mock_cfg_classes = mocker.patch.object(SegStage, "configure_classes")
        mock_cfg_ignore = mocker.patch.object(SegStage, "configure_ignore")
        self.stage.configure_task(self.stage.cfg, True)

        mock_cfg_classes.assert_called_once()
        mock_cfg_ignore.assert_called_once()

    @e2e_pytest_unit
    def test_configure_classes_replace(self, mocker):
        mocker.patch.object(Stage, "get_data_classes", return_value=["foo", "bar"])
        self.stage.configure_classes(self.stage.cfg, "REPLACE")

        assert "background" in self.stage.model_classes
        assert self.stage.model_classes == ["background", "foo", "bar"]

    @e2e_pytest_unit
    def test_configure_classes_merge(self, mocker):
        mocker.patch.object(Stage, "get_model_classes", return_value=["foo", "bar"])
        mocker.patch.object(Stage, "get_data_classes", return_value=["foo", "baz"])
        self.stage.configure_classes(self.stage.cfg, "MERGE")

        assert "background" in self.stage.model_classes
        assert self.stage.model_classes == ["background", "foo", "bar", "baz"]

    @e2e_pytest_unit
    def test_configure_ignore(self):
        self.stage.configure_ignore(self.stage.cfg)

        if "decode_head" in self.stage.cfg.model:
            assert self.stage.cfg.model.decode_head.loss_decode.type == "CrossEntropyLossWithIgnore"
