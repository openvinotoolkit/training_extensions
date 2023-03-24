import pytest

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.mpa.det.semisl.stage import SemiSLDetectionStage
from otx.mpa.det.stage import DetectionStage
from tests.test_suite.e2e_test_system import e2e_pytest_unit

SEMISL_RECIPE_CONFIG_PATH = "otx/recipes/stages/detection/semisl.py"


class TestSemiSLDetectionStage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(SEMISL_RECIPE_CONFIG_PATH)
        self.stage = SemiSLDetectionStage(name="", mode="train", config=cfg, common_cfg=None, index=0)

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        mock_ul_dataloader = mocker.patch.object(DetectionStage, "configure_unlabeled_dataloader")
        fake_semisl_data_cfg = {"data": {"unlabeled": {"otx_dataset": "foo"}}}
        self.stage.configure_data(self.stage.cfg, True, fake_semisl_data_cfg)
        mock_ul_dataloader.assert_called_once()

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        fake_model_cfg = {"model": {"type": "", "task_adapt": True}}
        self.stage.cfg.merge_from_dict(fake_model_cfg)
        mock_config_bbox_head = mocker.patch.object(SemiSLDetectionStage, "configure_bbox_head")
        self.stage.configure_task(self.stage.cfg, True)
        mock_config_bbox_head.assert_called_once()
