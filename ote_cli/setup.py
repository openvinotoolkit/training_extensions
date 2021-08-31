import os

from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as read_file:
    requirements = [requirement.strip() for requirement in read_file]

setup(
    name='ote_cli',
    version='0.2',
    packages=find_packages(exclude=('tools',)),
    install_requires=requirements,
        entry_points={
            "console_scripts": [
                "ote_train=ote_cli.tools.train:main",
                "ote_eval=ote_cli.tools.eval:main",
                "ote_export=ote_cli.tools.export:main",
                "ote_optimize=ote_cli.tools.optimize:main",
            ]
        },
)
