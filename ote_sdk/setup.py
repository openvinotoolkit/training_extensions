"""
setup file for OTE SDK
"""

# INTEL CONFIDENTIAL
# Copyright (c) 2021 Intel Corporation
#
# The source code contained or described herein and all documents related to
# the source code ("Material") are owned by Intel Corporation or its suppliers
# or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary
# and confidential information of Intel or its suppliers and licensors. The
# Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified,
# published, uploaded, posted, transmitted, distributed, or disclosed in any way
# without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery
# of the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.

from setuptools import find_packages, setup

install_requires = []

with open("requirements.txt", "r", encoding="UTF-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            install_requires.append(line)

setup(
    name="OTE SDK",
    version="1.0",
    packages=find_packages(include=["ote_sdk", "ote_sdk.*"]),
    package_data={"ote_sdk": ["py.typed", "usecases/exportable_code/demo/*"]},
    url="",
    license="Copyright (c) Intel - All Rights Reserved. "
    "Unauthorized copying of any part of the software via any medium is strictly prohibited. "
    "Proprietary and confidential.",
    install_requires=install_requires,
    author="Intel",
    description="OTE SDK Package",
)
