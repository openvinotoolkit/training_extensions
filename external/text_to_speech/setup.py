import os.path
import re

import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages


# N.B.: Distribution root directory is the current working directory (CWD) because:
# 1) pip invokes setup.py with cwd=unpacked_source_directory and
# 2) find_packages() without args looks for packages in CWD.

def readme():
    with open('README.md') as f:
        content = f.read()
    return content

def get_version():
    re_version = re.compile('^__version__[ \t]*=[ \t]*[\'"](?P<version>[^\'"\\\\]*)[\'"][ \t]*(#.*)?$')
    with open(os.path.join('torchtts', 'version.py'), 'rt') as ver_file:
        for line in ver_file:
            m_version = re_version.match(line.rstrip())
            if m_version:
                return m_version.group('version')
    raise NameError("name '__version__' is not defined")

def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include

ext_modules = [
    Extension(
        'monotonic_align',
        [os.path.join('monotonic_align', 'core.pyx')],
        include_dirs=[numpy_include()],
    )
]

def get_requirements():
    requirements = []
    with open('requirements.txt', 'rt') as req_file:
        for line in req_file.readlines():
            line = line.rstrip()
            if line != '':
                requirements.append(line)
    return requirements

setup(
    name='torchtts',
    version=get_version(),
    description='An OpenVINO Training Extensions backend to train text-to-speech model with PyTorch',
    author='Intel Corporation',
    license='Apache-2.0',
    long_description=readme(),
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    install_requires=get_requirements(),
    keywords=['Text To Speech', 'Deep Learning', 'NLP'],
)
