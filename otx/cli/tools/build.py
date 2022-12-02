"""OTX building command 'otx build'.

This command allows you to build an OTX workspace, provide usable backbone configurations,
and build models with new backbone replacements.
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
import os

from otx.cli.builder import Builder

SUPPORTED_TASKS = ("CLASSIFICATION", "DETECTION", "INSTANCE_SEGMENTATION", "SEGMENTATION")


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help=f"The currently supported options: {SUPPORTED_TASKS}.")
    parser.add_argument("--workspace-root", help="The path to use as the workspace.")
    parser.add_argument("--model", help="Input OTX model config file (e.g model.py).", default=None)
    parser.add_argument("--backbone", help="Enter the backbone configuration file path or available backbone type.")
    parser.add_argument("--save-backbone-to", help="Enter where to save the backbone configuration file.", default=None)
    parser.add_argument("--root", help="A root dir where templates should be searched.", default=".")

    return parser.parse_args()


def main():
    """Main function for model or backbone or task building."""

    args = parse_args()

    builder = Builder()

    # Build with task_type -> Create User workspace
    if args.task and args.task.upper() in SUPPORTED_TASKS:
        builder.build_task_config(args.task, args.model, args.workspace_root, args.root)

    # Build Backbone related
    if args.backbone:
        missing_args = []
        if not args.backbone.endswith((".yml", ".yaml", ".json")):
            if args.save_backbone_to is None:
                args.save_backbone_to = (
                    os.path.join(args.workspace_root, "backbone.yaml") if args.workspace_root else "./backbone.yaml"
                )
            missing_args = builder.build_backbone_config(args.backbone, args.save_backbone_to)
            args.backbone = args.save_backbone_to
        if args.model:
            if missing_args:
                raise ValueError("The selected backbone has inputs that the user must enter.")
            builder.build_model_config(args.model, args.backbone)


if __name__ == "__main__":
    main()
