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

import os
from setuptools import setup

with open('./requirements.txt') as f:
    REQUIRED = f.read().splitlines()

# Add Model Optimizer requirements
requirements_onnx = os.path.join(os.environ.get('INTEL_OPENVINO_DIR', '/opt/intel/openvino'),
                                 './deployment_tools/model_optimizer/requirements_onnx.txt')
if os.path.isfile(requirements_onnx):
    with open(requirements_onnx) as f:
        REQUIRED += f.read().splitlines()

setup(
    name='segthor',
    install_requires=REQUIRED,
)
