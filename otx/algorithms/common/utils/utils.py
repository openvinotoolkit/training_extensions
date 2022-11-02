"""Collections of Dataset utils for common OTX algorithms."""

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

import importlib

import yaml

from otx.api.utils.argument_checks import YamlFilePathCheck, check_input_parameters_type


@check_input_parameters_type({"path": YamlFilePathCheck})
def load_template(path):
    """Loading model template function."""
    with open(path, encoding="UTF-8") as f:
        template = yaml.safe_load(f)
    return template


@check_input_parameters_type()
def get_task_class(path: str):
    """Return Task classes."""
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
