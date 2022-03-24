import os.path as osp
from setuptools import setup, Extension, find_packages

import numpy as np
from Cython.Build import cythonize

repo_root = osp.dirname(osp.realpath(__file__))

def readme():
    with open('README.md') as f:
        content = f.read()
    return content

def get_requirements(filename='requirements.txt'):
    here = osp.dirname(osp.realpath(__file__))
    requires = []
    links = []
    with open(osp.join(here, filename), 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if '-f http' in line:
                links.append(line)
            else:
                requires.append(line)
    return requires, links

packages, links = get_requirements()

setup(
    name='tts_demo',
    version="0.0.0",
    description='Text to speech demo',
    author='Intel Corporation',
    license='Apache-2.0',
    long_description=readme(),
    dependency_links=links,
    packages=find_packages(),
    install_requires=packages,
    keywords=['Text To Speech', 'Deep Learning', 'NLP'],
)
