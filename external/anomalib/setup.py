#!/usr/bin/env python
from setuptools import find_packages, setup

import anomalib

setup(
    name="anomaly_classification",
    version=anomalib.__version__,
    packages=find_packages(include=["anomaly_classification", "anomaly_classification.*"]),
    url="",
    license="license='Apache License 2.0'",
    install_requires="anomalib"
    author="Intel",
    description="anomaly classification - OpenVINO Training Extension for Anomaly Classification using anomalib library",
)
