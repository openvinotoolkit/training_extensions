import os
import pytest
import sys
from unittest.mock import patch

from ote_cli.registry import Registry
from ote_cli.utils.tests import (
    create_venv,
    get_some_vars,
)

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component


TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
root = "/tmp/ote_cli/"
templates = Registry("external/model-preparation-algorithm").filter(task_type="CLASSIFICATION").templates
templates_ids = [template.model_template_id for template in templates]


class TestToolsMPATelemetry:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, _, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @patch("ote_cli.tools.ote.ote_demo", return_value=None)
    @patch("ote_cli.utils.telemetry.init_telemetry_session", return_value=None)
    @patch("ote_cli.utils.telemetry.close_telemetry_session", return_value=None)
    @patch("ote_cli.utils.telemetry.send_version", return_value=None)
    @patch("ote_cli.utils.telemetry.send_cmd_results", return_value=None)
    def test_tm_integration_exit_0(self, mock_send_cmd, mock_send_version, mock_close_tm, mock_init_tm, mock_demo):
        from ote_cli.tools import ote

        backup_argv = sys.argv
        sys.argv = ["ote", "demo"]
        ret = ote.main()
        sys.argv = backup_argv

        assert ret == 0
        mock_demo.assert_called_once()
        mock_init_tm.assert_called_once()
        mock_close_tm.assert_called_once()
        mock_send_cmd.assert_called_with(None, "demo", {"retcode": 0})

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @patch("ote_cli.tools.ote.ote_demo", side_effect=Exception())
    @patch("ote_cli.utils.telemetry.init_telemetry_session", return_value=None)
    @patch("ote_cli.utils.telemetry.close_telemetry_session", return_value=None)
    @patch("ote_cli.utils.telemetry.send_version", return_value=None)
    @patch("ote_cli.utils.telemetry.send_cmd_results", return_value=None)
    def test_tm_integration_exit_exception(
        self,
        mock_send_cmd,
        mock_send_version,
        mock_close_tm,
        mock_init_tm,
        mock_demo,
    ):
        from ote_cli.tools import ote

        backup_argv = sys.argv
        sys.argv = ["ote", "demo"]
        with pytest.raises(Exception) as e:
            ote.main()
        sys.argv = backup_argv

        assert e.type == Exception, f"{e}"
        mock_init_tm.assert_called_once()
        mock_close_tm.assert_called_once()
        mock_send_cmd.assert_called_with(None, "demo", {"retcode": -1, "exception": repr(Exception())})
