import pytest

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.mpa.seg.semisl.stage import SemiSLSegStage
from otx.mpa.seg.stage import SegStage
from tests.test_suite.e2e_test_system import e2e_pytest_unit

SEMISL_RECIPE_CONFIG_PATH = "otx/recipes/stages/segmentation/semisl.py"


class TestOTXSegStage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(SEMISL_RECIPE_CONFIG_PATH)
        self.stage = SemiSLSegStage(name="", mode="train", config=cfg, common_cfg=None, index=0)

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        mock_ul_dataloader = mocker.patch.object(SegStage, "configure_unlabeled_dataloader")
        fake_semisl_data_cfg = {"data": {"unlabeled": {"otx_dataset": "foo"}}}
        self.stage.configure_data(self.stage.cfg, True, fake_semisl_data_cfg)

        mock_ul_dataloader.assert_called_once()

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        fake_model_cfg = {"model": {"type": "", "task_adapt": True}}
        self.stage.cfg.merge_from_dict(fake_model_cfg)
        mock_remove_hook = mocker.patch("otx.mpa.seg.semisl.stage.remove_custom_hook")
        self.stage.configure_task(self.stage.cfg, True)

        assert "task_adapt" not in self.stage.cfg.model
        mock_remove_hook.assert_called_once()
