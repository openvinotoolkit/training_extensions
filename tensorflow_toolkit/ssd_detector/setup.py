#!/usr/bin/env python3
#
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

# Under the Travis-CI we have to use a CPU version
if os.environ.get('CPU_ONLY', None) == 'true':
  REQUIRED = [p.replace('-gpu', '') if p.startswith('tensorflow') else p for p in REQUIRED]

setup(
  name='ssd_detector',
  version='0.2.2.dev0',
  install_requires=REQUIRED
)
