"""
Install anomalib wrapper for OTE
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

from typing import List

import anomalib
from setuptools import find_packages, setup


def get_requirements() -> List[str]:
    """Read requirements.txt and return the requirements.

    Returns:
        List[str]: List of requirements.
    """
    requirements = []
    with open("requirements.txt", "r", encoding="utf8") as file:
        for line in file.readlines():
            requirements.append(line.strip())
    return requirements


setup(
    name="anomaly_classification",
    version=anomalib.__version__,
    packages=find_packages(
        include=["anomaly_classification", "anomaly_classification.*", "ote_anomalib", "ote_anomalib.*"]
    ),
    url="",
    license="license='Apache License 2.0'",
    install_requires=get_requirements(),
    author="Intel",
    description="anomaly classification - "
    "OpenVINO Training Extension for Anomaly Classification using anomalib library",
)
