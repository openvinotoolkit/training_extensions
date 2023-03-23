import pytest

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.mpa.det.incremental.stage import IncrDetectionStage
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
@pytest.mark.parametrize("model_classes", [["red", "green"], ["red", "green", "blue"]])
def test_configure_task(model_classes, mocker):
    cfg = MPAConfig(dict(model=dict(type="FakeDetector"), task_adapt=dict()))
    stage = IncrDetectionStage(name="", mode="train", config=cfg, common_cfg=None, index=0)
    stage.task_adapt_type = "mpa"
    stage.org_model_classes = ["red", "green"]
    stage.model_classes = model_classes
    mock_config_task = mocker.patch("otx.mpa.det.stage.DetectionStage.configure_task")
    stage.configure_task(cfg, True)
    mock_config_task.assert_called_once()
