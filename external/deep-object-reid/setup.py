# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp

from setuptools import find_packages, setup

repo_root = osp.dirname(osp.realpath(__file__))

def get_requirements(filename):
    requires = []
    links = []
    with open(osp.join(repo_root, filename), 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if '-f http' in line:
                links.append(line)
            else:
                requires.append(line)
    return requires, links

requirements, links = get_requirements('requirements.txt')

setup(
    name='torchreid_tasks',
    packages=find_packages(),
    install_requires=requirements,
    dependency_links=links,
)
