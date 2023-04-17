import pytest

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.algorithms.segmentation.adapters.mmseg.tasks.incremental.stage import (
    IncrSegStage,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.segmentation.test_helpers import DEFAULT_RECIPE_CONFIG_PATH


class TestOTXSegStage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(DEFAULT_RECIPE_CONFIG_PATH)
        self.stage = IncrSegStage(name="", mode="train", config=cfg, common_cfg=None, index=0)
        self.stage.cfg.model["type"] = "ClassIncrEncoderDecoder"

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        mock_update_hook = mocker.patch(
            "otx.algorithms.segmentation.adapters.mmseg.tasks.incremental.stage.update_or_add_custom_hook"
        )
        self.stage.configure_task(self.stage.cfg, True)

        mock_update_hook.assert_called_once()
