import os

import pytest

from otx.algorithms.common.adapters.mmcv.tasks.exporter import Exporter
from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig
from otx.algorithms.common.adapters.mmdeploy.apis import NaiveExporter
from otx.algorithms.detection.adapters.mmdet.utils.builder import build_detector
from otx.algorithms.detection.adapters.mmdet.utils.exporter import DetectionExporter
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_RECIPE_CONFIG_PATH,
    DEFAULT_DET_TEMPLATE_DIR,
    DEFAULT_ISEG_RECIPE_CONFIG_PATH,
    DEFAULT_ISEG_TEMPLATE_DIR,
)


@e2e_pytest_unit
@pytest.mark.parametrize(
    "recipe_cfg, template_dir",
    [
        (DEFAULT_DET_RECIPE_CONFIG_PATH, DEFAULT_DET_TEMPLATE_DIR),
        (DEFAULT_ISEG_RECIPE_CONFIG_PATH, DEFAULT_ISEG_TEMPLATE_DIR),
    ],
)
def test_run(recipe_cfg, template_dir, mocker):
    exporter = DetectionExporter()
    model_cfg = OTXConfig.fromfile(os.path.join(template_dir, "model.py"))
    model_cfg.work_dir = "/tmp/"
    args = {"precision": "FP32", "model_builder": build_detector}
    mocker.patch.object(Exporter, "run", return_value=True)
    returned_value = exporter.run(model_cfg)
    assert "model_builder" in args
    assert returned_value is True


@e2e_pytest_unit
@pytest.mark.parametrize(
    "recipe_cfg, template_dir",
    [
        (DEFAULT_DET_RECIPE_CONFIG_PATH, DEFAULT_DET_TEMPLATE_DIR),
        (DEFAULT_ISEG_RECIPE_CONFIG_PATH, DEFAULT_ISEG_TEMPLATE_DIR),
    ],
)
def test_naive_export(recipe_cfg, template_dir, mocker):
    exporter = DetectionExporter()
    data_cfg = OTXConfig.fromfile(os.path.join(template_dir, "data_pipeline.py"))
    mock_export_ov = mocker.patch.object(NaiveExporter, "export2backend")
    exporter.naive_export("", build_detector, "FP32", "OPENVINO", data_cfg)
    mock_export_ov.assert_called_once()
