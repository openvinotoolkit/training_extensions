# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# flake8: noqa
import os

from .builder import build, build_workflow_hook
from .stage import Stage, get_available_types
from .version import __version__, get_version
from .workflow import Workflow


class MPAConstants:
    PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # PACKAGE_ROOT = os.path.dirname(Path(__file__).)
    RECIPES_PATH = os.path.join(PACKAGE_ROOT, "recipes")
    SAMPLES_PATH = os.path.join(PACKAGE_ROOT, "samples")
    MODELS_PATH = os.path.join(PACKAGE_ROOT, "models")


# print(f'pkg root ======> {MPAConstants.PACKAGE_ROOT}')

__all__ = [get_version, __version__, build, build_workflow_hook, Stage, get_available_types, Workflow, MPAConstants]
