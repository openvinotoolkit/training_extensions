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

SUPPORTED_TASKS = ("CLASSIFICATION", "DETECTION", "INSTANCE_SEGMENTATION", "SEGMENTATION")


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help=f"The currently supported options: {SUPPORTED_TASKS}.")
    parser.add_argument("--model", help="Input OTX model config file (e.g model.py).", default=None)
    parser.add_argument("--backbone", help="Enter the backbone configuration file path or available backbone type.")
    parser.add_argument("--save-backbone-to", default="./backbone.yaml")

    return parser.parse_args()


def main():
    """Main function for model templates searching."""

    args = parse_args()

    builder = Builder()

    # Build with task_type -> Create User workspace
    if args.task and args.task.lower() in SUPPORTED_TASKS:
        builder.build_task_config(args.task, args.model)

    # Build Backbone related
    if args.backbone:
        missing_args = []
        if not args.backbone.endswith((".yml", ".yaml", ".json")):
            missing_args = builder.build_backbone_config(args.backbone, args.save_backbone_to)
            args.backbone = args.save_backbone_to
        if args.model:
            if missing_args:
                raise ValueError("The selected backbone has inputs that the user must enter.")
            builder.build_model_config(args.model, args.backbone)


if __name__ == "__main__":
    main()
