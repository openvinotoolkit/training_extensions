"""
OTE CLI entry point.
"""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import sys

from .eval import main as ote_eval
from .export import main as ote_export
from .find import main as ote_find
from .train import main as ote_train

__all__ = ["ote_eval", "ote_export", "ote_find", "ote_train"]


def parse_args():
    """
    Parses command line arguments.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("operation", choices=["find", "train", "eval", "export"])

    return parser.parse_known_args()[0]


def main():
    """
    This function is a single entry point for all OTE CLI related operations:
      - train
      - eval
      - export
      - find
    """

    name = parse_args().operation
    sys.argv[0] = f"ote {name}"
    del sys.argv[1]
    globals()[f"ote_{name}"]()
