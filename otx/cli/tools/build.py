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
from otx.api.entities.model_template import ModelTemplate

SUPPORTED_TASKS = ("CLASSIFICATION", "DETECTION", "INSTANCE_SEGMENTATION", "SEGMENTATION")
SUPPORTED_TRAIN_TYPE = ("incremental", "semisl", "selfsl")


def set_workspace(task: str, model: str, root: str ="", name: str ="otx-workspace"):
    """Set workspace path according to arugments."""
    path = f"{root}/{name}-{task}" if root != "" else f"./{name}-{task}"
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

def build(
    builder: Builder,
    train_data_roots: str = None,
    val_data_roots: str = None,
    task: str = None,
    train_type: str = "incremental",
    workspace_root: str = None,
    model: str = None,
    backbone: str = None,
    save_backbone_to: str = None,
    otx_root: str = ".",
    template: ModelTemplate = None,
):
    
    # Auto-configuration
    config_manager = ConfigManager()
    if train_data_roots:
        if task is None:
            task_type = config_manager.auto_task_detection(train_data_roots)
            task = task_type
        if val_data_roots is None:
            config_manager.auto_split_data(train_data_roots, task)
    
    if task and task in SUPPORTED_TASKS:
        # Build with task_type and create user workspace
        if workspace_root is None:
            workspace_root = set_workspace(task, model)
        builder.build_task_config(
            task_type=task,
            model_type=model,
            train_type=train_type.lower(),
            workspace_path=Path(workspace_root),
            otx_root=otx_root,
            template=template
        )
        config_manager.write_data_with_cfg(
            workspace_dir=workspace_root,
            train_data_roots=train_data_roots,
            val_data_roots=val_data_roots,
        )

    # Build Backbone related
    if backbone:
        missing_args = []
        if not backbone.endswith((".yml", ".yaml", ".json")):
            if save_backbone_to is None:
                save_backbone_to = (
                    os.path.join(workspace_root, "backbone.yaml") if workspace_root else "./backbone.yaml"
                )
            missing_args = builder.build_backbone_config(backbone, save_backbone_to)
            backbone = save_backbone_to
        if model:
            if missing_args:
                raise ValueError("The selected backbone has inputs that the user must enter.")
            builder.merge_backbone(model, backbone)

def main():
    """Main function for model or backbone or task building."""

    args = parse_args()
    args.task = args.task.upper() if args.task is not None else args.task

    builder = Builder()
    otx_root = get_otx_root_path()

    build(
        builder=builder,
        train_data_roots=args.train_data_roots,
        val_data_roots=args.val_data_roots,
        task=args.task,
        train_type=args.train_type,
        workspace_root=args.workspace_root,
        model=args.model,
        backbone=args.backbone,
        save_backbone_to=args.save_backbone_to,
        otx_root=otx_root
    )

if __name__ == "__main__":
    main()
