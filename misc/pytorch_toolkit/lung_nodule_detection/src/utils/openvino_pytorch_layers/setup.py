#!/usr/bin/env python
import os
from setuptools import setup

if not 'VERSION' in os.environ:
    raise Exception('Specify package version by <VERSION> environment variable')

if not 'EXT_LIB' in os.environ:
    raise Exception('Specify <EXT_LIB> environment variable with a path to extensions library')

setup(name='openvino-extensions',
      version=os.environ['VERSION'],
      author='Dmitry Kurtaev',
      url='https://github.com/dkurt/openvino_pytorch_layers',
      packages=['openvino_extensions'],
      data_files=[('../../openvino_extensions', [os.environ['EXT_LIB']])],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
      ],
)
