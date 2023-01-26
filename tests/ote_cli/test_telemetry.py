import sys
import subprocess
import pytest
import unittest
from unittest.mock import MagicMock, patch

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component

from ote_cli.tools import ote

from ote_cli.tools.ote import (
    ote_demo,
    ote_deploy,
    ote_eval,
    ote_export,
    ote_find,
    ote_optimize,
    ote_train
)


class TestTelemetry(unittest.TestCase):

    @e2e_pytest_component
    @patch("ote_cli.tools.ote.ote_demo", return_value=None)
    @patch("ote_cli.utils.telemetry.init_telemetry_session", return_value=None)
    @patch("ote_cli.utils.telemetry.close_telemetry_session", return_value=None)
    @patch("ote_cli.utils.telemetry.send_version", return_value=None)
    @patch("ote_cli.utils.telemetry.send_cmd_results", return_value=None)
    def test_tm_integration_exit_0(self,
        mock_send_cmd,
        mock_send_version,
        mock_close_tm,
        mock_init_tm,
        mock_demo
    ):
        backup_argv = sys.argv
        sys.argv = ["ote", "demo"]
        ret = ote.main()
        sys.argv = backup_argv

        self.assertEqual(ret, 0)
        mock_demo.assert_called_once()
        mock_init_tm.assert_called_once()
        mock_close_tm.assert_called_once()
        mock_send_cmd.assert_called_with(None, "demo", {"retcode": 0})

    @e2e_pytest_component
    @patch("ote_cli.tools.ote.ote_demo", side_effect=Exception())
    @patch("ote_cli.utils.telemetry.init_telemetry_session", return_value=None)
    @patch("ote_cli.utils.telemetry.close_telemetry_session", return_value=None)
    @patch("ote_cli.utils.telemetry.send_version", return_value=None)
    @patch("ote_cli.utils.telemetry.send_cmd_results", return_value=None)
    def test_tm_integration_exit_exception(self,
        mock_send_cmd,
        mock_send_version,
        mock_close_tm,
        mock_init_tm,
        mock_demo,
    ):
        backup_argv = sys.argv
        sys.argv = ["ote", "demo"]
        with pytest.raises(Exception) as e:
            ret = ote.main()
        sys.argv = backup_argv

        self.assertEqual(e.type, Exception, f"{e}")
        mock_init_tm.assert_called_once()
        mock_close_tm.assert_called_once()
        mock_send_cmd.assert_called_with(None, "demo", {"retcode": -1, "exception": repr(Exception())})
