# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

from pathlib import Path
from setuptools import setup, find_packages


SETUP_DIR = Path(__file__).resolve().parent

with open(SETUP_DIR / 'requirements.txt') as f:
    required = f.read().splitlines()

packages = find_packages(str(SETUP_DIR))
package_dir = {packages[0]: str(SETUP_DIR / packages[0])}

setup(
    name=packages[0],
    version='0.0',
    author='Intel® Corporation',
    license="Copyright (c) Intel - All Rights Reserved. "
    "Unauthorized copying of any part of the software via any medium is strictly prohibited. "
    "Proprietary and confidential.",
    description='Demo based on ModelAPI classes',
    packages=packages,
    package_dir=package_dir,
    package_data={
        packages[0]: ['*.xml', '*.bin', '*.json'],
    },
    install_requires=required,
    entry_points={
        "console_scripts": ["{}={}.sync:main".format(packages[0], packages[0])]},
)
