"""
Install anomalib wrapper for OTE
"""

import anomalib
from setuptools import find_packages, setup

setup(
    name="anomaly_classification",
    version=anomalib.__version__,
    packages=find_packages(
        include=["anomaly_classification", "anomaly_classification.*", "ote_anomalib", "ote_anomalib.*"]
    ),
    url="",
    license="license='Apache License 2.0'",
    install_requires="anomalib",
    author="Intel",
    description="anomaly classification - "
    "OpenVINO Training Extension for Anomaly Classification using anomalib library",
)
