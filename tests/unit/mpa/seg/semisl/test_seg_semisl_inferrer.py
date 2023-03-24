import os

import pytest

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.mpa.seg.semisl.inferrer import SemiSLSegInferrer
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.segmentation.test_helpers import DEFAULT_SEG_TEMPLATE_DIR

SEMISL_RECIPE_CONFIG_PATH = "otx/recipes/stages/segmentation/semisl.py"


class TestOTXSegStage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(SEMISL_RECIPE_CONFIG_PATH)
        self.inferrer = SemiSLSegInferrer(name="", mode="train", config=cfg, common_cfg=None, index=0)
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "semisl/model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "semisl/data_pipeline.py"))

    @e2e_pytest_unit
    def test_configure(self):
        updated_cfg = self.inferrer.configure(self.model_cfg, "", self.data_cfg)

        assert "orig_type" not in updated_cfg
        assert "unsup_weight" not in updated_cfg
        assert "semisl_start_iter" not in updated_cfg
