"""Model templates searching tool."""

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

from otx.cli.builder import Builder


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Input OTX model config file (e.g model.py).", default=None)
    parser.add_argument("--backbone", help="Input backbone config file (e.g backbone.yaml).")
    parser.add_argument("--task_type", help="Input task type (e.g detection).")
    parser.add_argument("--save-to", default="./backbone.yaml")

    return parser.parse_args()


def main():
    """Main function for model templates searching."""

    args = parse_args()

    builder = Builder()
    if args.task_type:
        builder.build_task_config(args.task_type, args.model)
    if args.model and args.backbone:
        builder.build_model_config(args.model, args.backbone)
    elif args.backbone:
        builder.build_backbone_config(args.backbone, args.save_to)


if __name__ == "__main__":
    main()
