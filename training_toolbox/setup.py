import os
from setuptools import setup

with open('./requirements.txt') as f:
  REQUIRED = f.read().splitlines()

# Under the Travis-CI we have to use a CPU version
if os.environ.get('TRAVIS', None) == 'true':
  REQUIRED = [p.replace('-gpu', '') if p.startswith('tensorflow') else p for p in REQUIRED]

setup(
  name='Training toolbox',
  version='0.1.2.dev0',
  install_requires=REQUIRED
)
