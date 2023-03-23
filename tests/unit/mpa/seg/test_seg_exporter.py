import os

import pytest

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.algorithms.common.adapters.mmdeploy.apis import NaiveExporter
from otx.algorithms.segmentation.adapters.mmseg.utils.builder import build_segmentor
from otx.mpa.exporter_mixin import ExporterMixin
from otx.mpa.seg.exporter import SegExporter
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.segmentation.test_helpers import (
    DEFAULT_RECIPE_CONFIG_PATH,
    DEFAULT_SEG_TEMPLATE_DIR,
)


class TestOTXSegExporter:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(DEFAULT_RECIPE_CONFIG_PATH)
        self.exporter = SegExporter(name="", mode="train", config=cfg, common_cfg=None, index=0)
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_run(self, mocker):
        args = {"precision": "FP32", "model_builder": build_segmentor}
        mocker.patch.object(ExporterMixin, "run", return_value=True)
        returned_value = self.exporter.run(self.model_cfg, "", self.data_cfg, **args)

        assert "model_builder" in args
        assert returned_value is True

    @e2e_pytest_unit
    def test_naive_export(self, mocker):
        mock_export_ov = mocker.patch.object(NaiveExporter, "export2openvino")
        self.exporter.naive_export("", build_segmentor, "FP32", self.data_cfg)

        mock_export_ov.assert_called_once()
