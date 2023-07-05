"""Utilities for OpenVINO telemetry."""
# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=broad-exception-caught
import json

import openvino_telemetry as tm

from otx import __version__

__TM_CATEGORY_OTX = "otx"
__TM_MEASUREMENT_ID = "UA-17808594-29"
# __TM_MEASUREMENT_ID_FOR_TESTING = "UA-254359572-1"
# __TM_MEASUREMENT_ID = __TM_MEASUREMENT_ID_FOR_TESTING

__TM_ACTION_VERSION = "version"
__TM_ACTION_CMD_SUCCESS = "success"
__TM_ACTION_CMD_FAILURE = "failure"
__TM_ACTION_CMD_EXCEPTION = "exception"
__TM_ACTION_ERROR = "error"


def init_telemetry_session():
    """Init session."""
    telemetry = tm.Telemetry(app_name=__TM_CATEGORY_OTX, app_version=str(__version__), tid=__TM_MEASUREMENT_ID)
    telemetry.start_session(__TM_CATEGORY_OTX)
    send_version(telemetry)

    return telemetry


def close_telemetry_session(telemetry):
    """Close session."""
    if not isinstance(telemetry, tm.Telemetry):
        raise RuntimeError(f"Invalid argument. required {type(tm.Telemetry)} but passed {type(telemetry)}")
    telemetry.end_session(__TM_CATEGORY_OTX)
    telemetry.force_shutdown(1.0)


def send_version(telemetry):
    """Send application version."""
    if not isinstance(telemetry, tm.Telemetry):
        raise RuntimeError(f"Invalid argument. required {type(tm.Telemetry)} but passed {type(telemetry)}")
    __send_event(telemetry, __TM_ACTION_VERSION, str(__version__))


def send_cmd_results(telemetry, cmd, results):
    """Send cli telemetry data."""
    if not isinstance(telemetry, tm.Telemetry):
        raise RuntimeError(f"Invalid argument. required {type(tm.Telemetry)} but passed {type(telemetry)}")
    action = __TM_ACTION_ERROR

    if not isinstance(results, dict):
        raise RuntimeError(f"Invalid argument. required {dict} but passed {type(results)}")

    retcode = results.pop("retcode", None)
    if retcode is not None:
        label = dict(cmd=cmd, **results)
        if retcode >= 0:
            action = __TM_ACTION_CMD_FAILURE
            if retcode == 0:
                action = __TM_ACTION_CMD_SUCCESS
        else:
            action = __TM_ACTION_CMD_EXCEPTION
            label = dict(cmd=cmd, **results)

    if action == __TM_ACTION_ERROR:
        __send_error(telemetry, f"Invalid results for sending cmd result: {results}")
    else:
        __send_event(telemetry, action, label)


def __send_event(telemetry, action, label, **kwargs):
    """Wrapper of the openvino-telemetry.send_event()."""
    try:
        telemetry.send_event(__TM_CATEGORY_OTX, action, json.dumps(label), **kwargs)
    except Exception as error:
        print(f"An error while calling otm.send_event(): \n{repr(error)}")


def __send_error(telemetry, err_msg, **kwargs):
    """Wrapper of the openvino-telemetry.send_error()."""
    try:
        telemetry.send_error(__TM_CATEGORY_OTX, err_msg, **kwargs)
    except Exception as error:
        print(f"An error while calling otm.send_error(): \n{repr(error)}")
