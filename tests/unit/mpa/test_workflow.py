import mmcv

from otx.mpa.stage import Stage
from otx.mpa.workflow import Workflow
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestWorkflow:
    @e2e_pytest_unit
    def test_run(self, mocker):
        fake_cfg = {
            "work_dir": "/path/workdir",
            "seed": 0,
        }
        fake_common_cfg = {"output_path": "/path/output"}
        mocker.patch.object(mmcv, "mkdir_or_exist")
        stage = Stage(
            "MockStage",
            "",
            fake_cfg,
            fake_common_cfg,
            0,
            input=dict(
                arg_name1=dict(stage_name="MockStage", output_key=""),
                arg_name2=dict(stage_name="MockStage", output_key=""),
            ),
        )
        workflow = Workflow([stage])

        mock_stage_run = mocker.patch.object(stage, "run")
        workflow.run()

        mock_stage_run.assert_called()
