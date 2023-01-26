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
from pathlib import Path

from otx.cli.builder import Builder
from otx.cli.utils.importing import get_otx_root_path
from otx.core.config.manager import ConfigManager

SUPPORTED_TASKS = ("CLASSIFICATION", "DETECTION", "INSTANCE_SEGMENTATION", "SEGMENTATION")
SUPPORTED_TRAIN_TYPE = ("incremental", "semisl", "selfsl")


def set_workspace(task, model):
    """Set workspace path according to path, task, model arugments."""
    path = f"./otx-workspace-{task}"
    if model:
        path += f"-{model}"
    return path


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-roots", help="data root for training data", type=str, default=None)
    parser.add_argument("--val-data-roots", help="data root for validation data", type=str, default=None)
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

    args = parse_args()
    args.task = args.task.upper() if args.task is not None else args.task

    builder = Builder()

    otx_root = get_otx_root_path()

    # Auto-configuration
    config_manager = ConfigManager()
    if args.train_data_roots:
        if args.task is None:
            task_type = config_manager.auto_task_detection(args.train_data_roots)
            args.task = task_type
        if args.val_data_roots is None:
            config_manager.auto_split_data(args.train_data_roots, args.task)

    # Build with task_type and create user workspace
    if args.workspace_root is None:
        args.workspace_root = set_workspace(args.task, args.model)
    if args.task and args.task in SUPPORTED_TASKS:
        builder.build_task_config(
            task_type=args.task,
            model_type=args.model,
            train_type=args.train_type.lower(),
            workspace_path=Path(args.workspace_root),
            otx_root=otx_root,
        )

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
            builder.merge_backbone(args.model, args.backbone)

    config_manager.write_data_with_cfg(
        workspace_dir=args.workspace_root,
        train_data_roots=args.train_data_roots,
        val_data_roots=args.val_data_roots,
    )


if __name__ == "__main__":
    main()
