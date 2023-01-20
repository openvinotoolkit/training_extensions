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
from otx.core.data.manager.dataset_manager import DatasetManager

SUPPORTED_TASKS = ("CLASSIFICATION", "DETECTION", "INSTANCE_SEGMENTATION", "SEGMENTATION")
SUPPORTED_TRAIN_TYPE = ("incremental", "semisl", "selfsl")

def set_workspace(path, task, model):
    """Set workspace path according to path, task, model arugments."""
    if path is None:
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

    # Auto configuration & Datumaro Helper
    # First, find the format of train dataset and import it by using DatasetManager
    # Second, find the task type
    # Third, check the whether val_data_roots is None or not None
    # If val_data_roots is None, auto_split() is trigerred.
    # Forth, save the dataset with it's configuraiton(.yaml) file
    is_autoconfig_enabled = args.train_data_roots
    if is_autoconfig_enabled:
        print("[*] Auto-configuration is enabled !!")
        train_data_format = DatasetManager.get_data_format(args.train_data_roots)
        print(f"[*] Train data format: {train_data_format}")

        train_datumaro_dataset = DatasetManager.import_dataset(
            data_root=args.train_data_roots, data_format=train_data_format
        )

        splitted_dataset = {}
        splitted_dataset["train"] = DatasetManager.get_train_dataset(train_datumaro_dataset)

        config_manager = ConfigManager()
        train_task_type = config_manager.get_task_type(train_data_format)
        print(f"[*] Train task type: {train_task_type}")
        if args.task:
            assert (
                args.task == train_task_type
            ), f"Dataset format({train_data_format}) can't be used for {args.task} task."

        # Overwrite the args.task to train_task_type to select default template
        args.task = train_task_type if args.task is None else args.task

        # Create workspace
        args.workspace_root = set_workspace(args.workspace_root, args.task, args.model) 
        Path(args.workspace_root).mkdir(exist_ok=False)
        print(f"[*] Create workspace to: {args.workspace_root}")

        # If no validation dataset, auto_split will be triggered
        if args.val_data_roots is not None:
            val_data_format = DatasetManager.get_data_format(args.val_data_roots)
            val_datumaro_dataset = DatasetManager.import_dataset(
                data_root=args.val_data_roots, data_format=val_data_format
            )
            splitted_dataset["val"] = DatasetManager.get_val_dataset(val_datumaro_dataset)
        else:
            # TODO: consider automatic validation import i.e. COCO
            # Currently, automatic import will be ignored
            splitted_dataset = DatasetManager.auto_split(
                task=train_task_type, dataset=splitted_dataset["train"], split_ratio=[("train", 0.8), ("val", 0.2)]
            )

        # Will save the spliited dataset to workspace with .yaml file
        # For the classification task, imagenet_text format will be used to save the data
        # Also, data.yaml will be saved
        config_manager.write_data_with_cfg(splitted_dataset, train_data_format, args.workspace_root)

    args.workspace_root = set_workspace(args.workspace_root, args.task, args.model) 
    
    # Build with task_type and create user workspace
    if args.task and args.task in SUPPORTED_TASKS:
        builder.build_task_config(
            task_type=args.task,
            model_type=args.model,
            train_type=args.train_type.lower(),
            workspace_path=args.workspace_root,
            otx_root=otx_root,
            exist=is_autoconfig_enabled,
        )

    # Build Backbone related
    if args.backbone:
        missing_args = []
        if not args.backbone.endswith((".yml", ".yaml", ".json")):
            if args.save_backbone_to is None:
                args.save_backbone_to = os.path.join(args.workspace_root,"backbone.yaml") if args.workspace_root else "./backbone.yaml"
            missing_args = builder.build_backbone_config(args.backbone, args.save_backbone_to)
            args.backbone = args.save_backbone_to
        if args.model:
            if missing_args:
                raise ValueError("The selected backbone has inputs that the user must enter.")
            builder.merge_backbone(args.model, args.backbone)


if __name__ == "__main__":
    main()
