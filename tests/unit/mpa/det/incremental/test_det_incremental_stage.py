import pytest

from otx.mpa.det.incremental.stage import IncrDetectionStage
from otx.mpa.utils.config_utils import MPAConfig
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import DEFAULT_DET_RECIPE_CONFIG_PATH


class TestIncrDetectionStage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(DEFAULT_DET_RECIPE_CONFIG_PATH)
        self.stage = IncrDetectionStage(name="", mode="train", config=cfg, common_cfg=None, index=0)

    @e2e_pytest_unit
    def test_init(self):
        pass
