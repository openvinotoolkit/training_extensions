import os.path as osp
from setuptools import setup, Extension, find_packages

import numpy as np
from Cython.Build import cythonize

repo_root = osp.dirname(osp.realpath(__file__))

def readme():
    with open('README.md') as f:
        content = f.read()
    return content


def find_version():
    version_file = 'torchtts/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def numpy_include():
    print("NUMPY version: ", np.__version__)
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include

ext_modules = [
    Extension(
        'monotonic_align',
        [osp.join(repo_root, 'monotonic_align/core.pyx')],
        include_dirs=[numpy_include()],
    )
]

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
    name='torchtts',
    version=find_version(),
    description='A library for deep learning text to speech in PyTorch',
    author='Intel Corporation',
    license='Apache-2.0',
    long_description=readme(),
    dependency_links=links,
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    install_requires=packages,
    keywords=['Text To Speech', 'Deep Learning', 'NLP'],
)
