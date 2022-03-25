# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
from setuptools import setup, find_packages

repo_root = osp.dirname(osp.realpath(__file__))

def get_requirements(filename):
    requires = []
    with open(osp.join(repo_root, filename), 'r') as f:
        for line in f.readlines():
            requires.append(line.strip())
    return requires

requirements = get_requirements('requirements.txt')

setup(
    name='mpa_tasks',
    packages=find_packages(),
    install_requires=requirements,
)
