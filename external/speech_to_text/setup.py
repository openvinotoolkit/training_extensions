import os.path
import re

from setuptools import setup, find_packages


# N.B.: Distribution root directory is the current working directory (CWD) because:
# 1) pip invokes setup.py with cwd=unpacked_source_directory and
# 2) find_packages() without args looks for packages in CWD.

def get_version():
    re_version = re.compile('^__version__[ \t]*=[ \t]*[\'"](?P<version>[^\'"\\\\]*)[\'"][ \t]*(#.*)?$')
    with open(os.path.join('speech_to_text', 'version.py'), 'rt') as ver_file:
        for line in ver_file:
            m_version = re_version.match(line.rstrip())
            if m_version:
                return m_version.group('version')
    raise NameError("name '__version__' is not defined")

def get_requirements():
    requirements = []
    with open('requirements.txt', 'rt') as req_file:
        for line in req_file.readlines():
            line = line.rstrip()
            if line != '':
                requirements.append(line)
    return requirements

setup(
    name='speech_to_text',
    version=get_version(),
    description='An OpenVINO Training Extensions backend to train Speech To Text (ASR) models with PyTorch',
    author='Intel Corporation',
    license='Apache-2.0',
    packages=find_packages(),
    install_requires=get_requirements(),
    keywords=['Speech To Text', 'STT', 'Deep Learning', 'Speech Recognition'],
)
