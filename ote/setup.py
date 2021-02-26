import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as read_file:
    requirements = [requirement.strip() for requirement in read_file]

setup(
    name='ote',
    version='0.2',
    packages=find_packages(exclude=('tools',)),
    install_requires=requirements
)
