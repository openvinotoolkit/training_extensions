# nosec

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
Utilities for OpenVINO telemetry
"""

# pylint: disable=broad-except, ungrouped-imports
import json

from ote_cli import get_version

try:
    import openvino_telemetry as tm
except ImportError:
    from ote_cli.utils import telemetry_stub as tm


__TM_CATEGORY_OTX = "otx"
__TM_MEASUREMENT_ID = "UA-17808594-29"
# __TM_MEASUREMENT_ID_FOR_TESTING = "UA-254359572-1"

__TM_ACTION_VERSION = "version"
__TM_ACTION_CMD_SUCCESS = "success"
__TM_ACTION_CMD_FAILURE = "failure"
__TM_ACTION_CMD_EXCEPTION = "exception"
__TM_ACTION_ERROR = "error"


def init_telemetry_session():
    """init session"""
    telemetry = tm.Telemetry(
        app_name=__TM_CATEGORY_OTX, app_version=get_version(), tid=__TM_MEASUREMENT_ID
    )
    telemetry.start_session(__TM_CATEGORY_OTX)
    send_version(telemetry)

    return telemetry


def close_telemetry_session(telemetry):
    """close session"""
    telemetry.end_session(__TM_CATEGORY_OTX)
    telemetry.force_shutdown(1.0)


def send_version(telemetry):
    """send application version"""
    __send_event(telemetry, "version", str(get_version()))


def send_cmd_results(telemetry, cmd, results):
    """send cli telemetry data"""
    action = __TM_ACTION_ERROR
    retcode = results.pop("retcode", None)
    if retcode >= 0:
        action = __TM_ACTION_CMD_FAILURE
        if retcode == 0:
            action = __TM_ACTION_CMD_SUCCESS
        label = dict(cmd=cmd, **results)
    elif retcode < 0:
        action = __TM_ACTION_CMD_EXCEPTION
        label = dict(cmd=cmd, **results)

    __send_event(telemetry, action, label)


def __send_event(telemetry, action, label, **kwargs):
    """wrapper of the openvino-telemetry.send_event()"""
    if not isinstance(action, str):
        raise TypeError(f"action should string type but {type(action)}")
    if not isinstance(label, dict) and not isinstance(label, str):
        raise TypeError(f"label should 'dict' or 'str' type but {type(label)}")

    try:
        telemetry.send_event(__TM_CATEGORY_OTX, action, json.dumps(label), **kwargs)
    except Exception as error:
        print(f"An error while calling otm.send_event(): \n{repr(error)}")


def __send_error(telemetry, err_msg, **kwargs):
    """wrapper of the openvino-telemetry.send_error()"""
    if not isinstance(err_msg, str):
        raise TypeError(f"err_msg should string type but {type(err_msg)}")

    try:
        telemetry.send_error(__TM_CATEGORY_OTX, err_msg, **kwargs)
    except Exception as error:
        print(f"An error while calling otm.send_error(): \n{repr(error)}")
