# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
from setuptools import setup, Extension, find_packages

repo_root = osp.dirname(osp.realpath(__file__))

def find_version():
    version_file = osp.join(repo_root, 'speech_to_text/version.py')
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

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

packages, links = get_requirements('requirements.txt')

setup(
    name='speech_to_text',
    version=find_version(),
    description='A library for deep learning Speech To Text in PyTorch',
    author='Intel Corporation',
    license='Apache-2.0',
    dependency_links=links,
    packages=find_packages(),
    install_requires=packages,
    keywords=['Speech To Text', 'STT', 'Deep Learning', 'Speech Recognition'],
)
