from unittest.mock import MagicMock, patch

import openvino_telemetry as ovtm
import pytest

from otx import __version__
from otx.cli.utils.telemetry import (
    close_telemetry_session,
    init_telemetry_session,
    send_cmd_results,
    send_version,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestTelemetry:
    # FIXME: is there a way to get private constants for the testing?
    __TM_CATEGORY_OTX = "otx"
    __TM_MEASUREMENT_ID = "UA-17808594-29"
    __TM_ACTION_VERSION = "version"
    __TM_ACTION_CMD_SUCCESS = "success"
    __TM_ACTION_CMD_FAILURE = "failure"
    __TM_ACTION_CMD_EXCEPTION = "exception"
    __TM_ACTION_ERROR = "error"

    @e2e_pytest_unit
    @patch("otx.cli.utils.telemetry.tm")
    @patch("otx.cli.utils.telemetry.send_version")
    def test_init_telemetry_session(
        self,
        mock_send_version,
        mock_ovtm,
    ):
        mock_telemetry = MagicMock()
        mock_telemetry.start_session = MagicMock()
        mock_ovtm.Telemetry = MagicMock(return_value=mock_telemetry)

        init_telemetry_session()

        mock_ovtm.Telemetry.assert_called_once_with(
            app_name=self.__TM_CATEGORY_OTX, app_version=str(__version__), tid=self.__TM_MEASUREMENT_ID
        )
        mock_telemetry.start_session.assert_called_once_with(self.__TM_CATEGORY_OTX)
        mock_send_version.assert_called_once_with(mock_telemetry)

    @e2e_pytest_unit
    def test_close_telemetry_session(self):
        mock_ovtm_instance = MagicMock(spec=ovtm.Telemetry)

        close_telemetry_session(mock_ovtm_instance)

        mock_ovtm_instance.end_session.assert_called_once_with(self.__TM_CATEGORY_OTX)
        mock_ovtm_instance.force_shutdown.assert_called_once_with(1.0)

        with pytest.raises(RuntimeError):
            close_telemetry_session(0)

    @e2e_pytest_unit
    @patch("otx.cli.utils.telemetry.__send_event")
    def test_send_version(self, mock_send_event):
        mock_ovtm_instance = MagicMock(spec=ovtm.Telemetry)
        send_version(mock_ovtm_instance)
        mock_send_event.assert_called_once_with(mock_ovtm_instance, self.__TM_ACTION_VERSION, str(__version__))

        with pytest.raises(RuntimeError):
            send_version(None)

    @e2e_pytest_unit
    @patch("otx.cli.utils.telemetry.__send_event")
    @patch("otx.cli.utils.telemetry.__send_error")
    def test_send_cmd_results(
        self,
        mock_send_error,
        mock_send_event,
    ):
        with pytest.raises(RuntimeError):
            send_cmd_results(None, "something", None)

        mock_ovtm_instance = MagicMock(spec=ovtm.Telemetry)

        # with invalid results arg
        results = None

        with pytest.raises(RuntimeError):
            send_cmd_results(mock_ovtm_instance, "something", results)

        mock_send_error.reset_mock()
        mock_send_event.reset_mock()

        # with empty dict
        results = {}
        send_cmd_results(mock_ovtm_instance, "something", results)
        mock_send_error.assert_called_once()
        mock_send_event.assert_not_called()

        mock_send_error.reset_mock()
        mock_send_event.reset_mock()

        # with failure retcode
        results = {"retcode": 1, "some": "results"}
        cmd = "cmd"
        send_cmd_results(mock_ovtm_instance, cmd, results)
        mock_send_error.assert_not_called()
        mock_send_event.assert_called_once_with(
            mock_ovtm_instance, self.__TM_ACTION_CMD_FAILURE, dict(cmd=cmd, **results)
        )

        mock_send_error.reset_mock()
        mock_send_event.reset_mock()

        # with success retcode
        results = {"retcode": 0, "some": "results"}
        cmd = "cmd"
        send_cmd_results(mock_ovtm_instance, cmd, results)
        mock_send_error.assert_not_called()
        mock_send_event.assert_called_once_with(
            mock_ovtm_instance, self.__TM_ACTION_CMD_SUCCESS, dict(cmd=cmd, **results)
        )

        mock_send_error.reset_mock()
        mock_send_event.reset_mock()

        # with exception retcode
        results = {"retcode": -1, "some": "results"}
        cmd = "cmd"
        send_cmd_results(mock_ovtm_instance, cmd, results)
        mock_send_error.assert_not_called()
        mock_send_event.assert_called_once_with(
            mock_ovtm_instance, self.__TM_ACTION_CMD_EXCEPTION, dict(cmd=cmd, **results)
        )

        mock_send_error.reset_mock()
        mock_send_event.reset_mock()
