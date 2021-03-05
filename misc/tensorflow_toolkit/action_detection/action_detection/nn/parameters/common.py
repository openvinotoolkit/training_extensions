# Copyright (C) 2019 Intel Corporation
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

import yaml


class AttributedDict(dict):
    """Class to simplify the access to dictionary fields.
    """

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def load_config(config_path):
    """Loads parameters into the dict from the specified path.

    :param config_path: Path to config file
    :return: Dictionary with parameters
    """

    with open(config_path, 'r') as config_file:
        config_values = AttributedDict(yaml.load(config_file, Loader=yaml.FullLoader))

    return config_values
