import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as read_file:
    requirements = [requirement for requirement in read_file]

setup(
    name='ote',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements
)
