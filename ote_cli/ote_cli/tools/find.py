"""
Model templates searching tool.
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

from ote_cli.registry import Registry


def parse_args():
    """
    Parses command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="A root dir where templates should be searched.", default=".")
    parser.add_argument("--task_type")

    return parser.parse_args()


def main():
    """
    Main function for model templates searching.
    """

    args = parse_args()

    registry = Registry(args.root)
    if args.task_type:
        registry = registry.filter(task_type=args.task_type)

    print(registry)


if __name__ == "__main__":
    main()
