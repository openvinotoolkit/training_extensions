from setuptools import setup, Extension, find_packages

import numpy as np
from Cython.Build import cythonize


def readme():
    with open('README.md') as f:
        content = f.read()
    return content

def get_requirements():
    requirements = []
    with open('requirements.txt', 'rt') as req_file:
        for line in req_file.readlines():
            line = line.rstrip()
            if line != '':
                requirements.append(line)
    return requirements

setup(
    name='demo_package',
    version="0.0.0",
    description='Text to speech demo',
    author='Intel Corporation',
    license='Apache-2.0',
    long_description=readme(),
    packages=find_packages(),
    install_requires=get_requirements(),
    keywords=['Text To Speech', 'Deep Learning', 'NLP'],
)
