"""
 Copyright (c) 2020 Intel Corporation
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

import pkg_resources
import email
import os
NNCF_PACKAGE_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HW_CONFIG_RELATIVE_DIR = "hw_configs"


def get_install_type():
    try:
        d = pkg_resources.get_distribution('nncf').get_metadata('PKG-INFO')
    except pkg_resources.DistributionNotFound:
        # Working with NNCF while not installed as a package
        return "GPU"
    install_type = email.message_from_string(d)['Keywords']
    return install_type
