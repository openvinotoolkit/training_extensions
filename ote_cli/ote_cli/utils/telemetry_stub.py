# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
A stub classs of the OpenVINO telemetry which would be used when the telemetry
module is not installed.
"""


class Telemetry:
    """
    A stub for the Telemetry class, which is used when the Telemetry class
    is not available.
    """

    def __init__(self, *arg, **kwargs):
        """__init__"""

    def send_event(self, *arg, **kwargs):
        """send_event"""

    def send_error(self, *arg, **kwargs):
        """send_error"""

    def start_session(self, *arg, **kwargs):
        """start_session"""

    def end_session(self, *arg, **kwargs):
        """end_session"""

    def force_shutdown(self, *arg, **kwargs):
        """force_shutdown"""

    def send_stack_trace(self, *arg, **kwargs):
        """send_stack_trace"""
