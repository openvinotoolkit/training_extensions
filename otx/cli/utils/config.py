"""
Utils for working with Configurable parameters
"""

# Copyright (C) 2021 Intel Corporation
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


def override_parameters(overrides, parameters):
    """
    Overrides parameters values by overrides.
    """

    allowed_keys = {"default_value", "value"}
    for k, val in overrides.items():
        if isinstance(val, dict):
            if k in parameters.keys():
                override_parameters(val, parameters[k])
            else:
                raise ValueError(f'The "{k}" is not in original parameters.')
        else:
            if k in allowed_keys:
                parameters[k] = val
            else:
                raise ValueError(f'The "{k}" is not in allowed_keys: {allowed_keys}')
