"""
 Copyright (c) 2019-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

class VersionAgnosticNames:
    RELU = "RELU"


class TorchOpInfo:
    def __init__(self, torch_version: str, version_agnostic_name: str):
        self.torch_version = torch_version
        self.version_agnostic_name = version_agnostic_name


OPERATOR_NAME_LOOKUP_TABLE = {
    "relu_"     : TorchOpInfo("1.1.0", VersionAgnosticNames.RELU),
    "relu"      : TorchOpInfo("unknown", VersionAgnosticNames.RELU)
}


def get_version_agnostic_name(version_specific_name: str):
    if version_specific_name not in OPERATOR_NAME_LOOKUP_TABLE:
        return version_specific_name

    return OPERATOR_NAME_LOOKUP_TABLE[version_specific_name].version_agnostic_name
