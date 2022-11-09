"""Monitors for deep_object_reid tasks."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from otx.api.utils.argument_checks import check_input_parameters_type


class StopCallback:
    """Stop callback class."""

    def __init__(self):
        self.stop_flag = False

    def stop(self):
        """Stop."""
        self.stop_flag = True

    def check_stop(self):
        """Check Stop."""
        return self.stop_flag

    def reset(self):
        """Reset."""
        self.stop_flag = False


class DefaultMetricsMonitor:
    """Default metric monitor class."""

    def __init__(self):
        self.metrics_dict = {}

    @check_input_parameters_type()
    def add_scalar(self, capture: str, value: float, timestamp: int):
        """Add scalar."""
        if capture in self.metrics_dict:
            self.metrics_dict[capture].append((timestamp, value))
        else:
            self.metrics_dict[capture] = [
                (timestamp, value),
            ]

    def get_metric_keys(self):
        """Get metric keys."""
        return self.metrics_dict.keys()

    @check_input_parameters_type()
    def get_metric_values(self, capture: str):
        """Get metric values."""
        return [item[1] for item in self.metrics_dict[capture]]

    @check_input_parameters_type()
    def get_metric_timestamps(self, capture: str):
        """Get metric timestamps."""
        return [item[0] for item in self.metrics_dict[capture]]

    def close(self):
        """Close."""
        pass  # pylint: disable=unnecessary-pass
