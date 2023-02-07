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
from pathlib import Path

from otx.cli.builder import Builder
from otx.cli.manager import ConfigManager

SUPPORTED_TASKS = ("CLASSIFICATION", "DETECTION", "INSTANCE_SEGMENTATION", "SEGMENTATION")
SUPPORTED_TRAIN_TYPE = ("incremental", "semisl", "selfsl")


def set_workspace(task, model):
    """Set workspace path according to path, task, model arugments."""
    path = f"./otx-workspace-{task}"
    if model:
        path += f"-{model}"
    return path


def get_args():
    """Parses command line arguments."""
    # TODO: Declaring pre_parser to get the template
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("template", nargs="?", default=None)
    parsed, _ = pre_parser.parse_known_args()
    template = parsed.template
    parser = argparse.ArgumentParser()
    if template and Path(template).is_file():
        parser.add_argument("template")
    else:
        parser.add_argument("--template", required=False)

    parser.add_argument(
        "--train-data-roots",
        help="Comma-separated paths to training data folders.",
    )
    parser.add_argument(
        "--val-data-roots",
        help="Comma-separated paths to validation data folders.",
    )
    parser.add_argument(
        "--unlabeled-data-roots",
        help="Comma-separated paths to unlabeled data folders",
    )
    parser.add_argument(
        "--unlabeled-file-list",
        help="Comma-separated paths to unlabeled file list",
    )
    parser.add_argument("--task", help=f"The currently supported options: {SUPPORTED_TASKS}.")
    parser.add_argument(
        "--train-type",
        help=f"The currently supported options: {SUPPORTED_TRAIN_TYPE}.",
        type=str,
        default="incremental",
    )
    parser.add_argument("--workspace-root", help="The path to use as the workspace.")
    parser.add_argument("--model", help="Input OTX model config file (e.g model.py).", default=None)
    parser.add_argument("--backbone", help="Enter the backbone configuration file path or available backbone type.")
    parser.add_argument("--save-backbone-to", help="Enter where to save the backbone configuration file.", default=None)

    return parser.parse_args()


def main():
    """Main function for model or backbone or task building."""

    args = get_args()
    config_manager = ConfigManager(args, mode="build")
    config_manager.task_type = args.task.upper() if args.task else ""
    config_manager.train_type = args.train_type if args.train_type else ""

    # Auto-Configuration for model template
    config_manager.configure_template()

    if not config_manager.check_workspace():
        new_workspace_path = None
        if args.workspace_root:
            new_workspace_path = args.workspace_root
        config_manager.build_workspace(new_workspace_path=new_workspace_path)

    # Auto-Configuration for Dataset configuration
    config_manager.configure_data_config()

    # Build Backbone related
    if args.backbone:
        builder = Builder()
        missing_args = []
        if not args.backbone.endswith((".yml", ".yaml", ".json")):
            if args.save_backbone_to is None:
                args.save_backbone_to = str(config_manager.workspace_root / "backbone.yaml")
            missing_args = builder.build_backbone_config(args.backbone, args.save_backbone_to)
            args.backbone = args.save_backbone_to
        if args.model:
            if missing_args:
                raise ValueError("The selected backbone has inputs that the user must enter.")
            builder.merge_backbone(args.model, args.backbone)


if __name__ == "__main__":
    main()
