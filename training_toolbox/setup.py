import os
from setuptools import setup

with open('./requirements.txt') as f:
  required = f.read().splitlines()

# Under the Travis we have to use a CPU version
if os.environ.get('TRAVIS', None) == 'true':
  required = [p.replace('-gpu', '') if p.startswith('tensorflow') else p for p in required]

setup(
    name='Training toolbox',
    version='0.1.2.dev0',
    install_requires=required
)
