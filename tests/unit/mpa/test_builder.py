import mmcv

from otx.mpa.builder import build
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_build_with_stages(mocker):
    cfg = mmcv.ConfigDict(
        stages=[mocker.MagicMock()],
        type=mocker.MagicMock(),
        workflow_hooks=[mocker.MagicMock()],
    )
    mocker.patch("otx.mpa.builder.build_workflow_hook")
    mock_build_from_cfg = mocker.patch("otx.mpa.builder.build_from_cfg")
    mock_workflow = mocker.patch("otx.mpa.builder.Workflow")
    mocker.patch("otx.mpa.builder.config_logger")
    mocker.patch("os.makedirs")
    mocker.patch("os.unlink")
    mocker.patch("os.symlink")

    build(cfg)

    mock_build_from_cfg.assert_called()
    mock_workflow.assert_called_once()


@e2e_pytest_unit
def test_build_without_stages(mocker):
    cfg = mmcv.ConfigDict()

    mocker.patch("otx.mpa.builder.get_available_types", return_value="MockStage")
    mock_build_from_cfg = mocker.patch("otx.mpa.builder.build_from_cfg")

    build(cfg, None, "MockStage")

    mock_build_from_cfg.assert_called_once()
