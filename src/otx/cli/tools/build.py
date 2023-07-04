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

from otx.cli.manager.config_manager import TASK_TYPE_TO_SUB_DIR_NAME, ConfigManager
from otx.cli.utils.parser import get_parser_and_hprams_data

SUPPORTED_TASKS = (
    "CLASSIFICATION",
    "DETECTION",
    "INSTANCE_SEGMENTATION",
    "ROTATED_DETECTION",
    "SEGMENTATION",
    "ACTION_CLASSIFICATION",
    "ACTION_DETECTION",
    "VISUAL_PROMPTING",
    "ANOMALY_CLASSIFICATION",
    "ANOMALY_DETECTION",
    "ANOMALY_SEGMENTATION",
)


def get_args():
    """Parses command line arguments."""
    parser, _, _ = get_parser_and_hprams_data()

    parser.add_argument(
        "--train-data-roots",
        help="Comma-separated paths to training data folders.",
    )
    parser.add_argument("--train-ann-files", help="Comma-separated paths to train annotation files.")
    parser.add_argument(
        "--val-data-roots",
        help="Comma-separated paths to validation data folders.",
    )
    parser.add_argument("--val-ann-files", help="Comma-separated paths to train annotation files.")
    parser.add_argument(
        "--test-data-roots",
        help="Comma-separated paths to test data folders.",
    )
    parser.add_argument("--test-ann-files", help="Comma-separated paths to train annotation files.")
    parser.add_argument(
        "--unlabeled-data-roots",
        help="Comma-separated paths to unlabeled data folders",
    )
    parser.add_argument(
        "--unlabeled-file-list",
        help="Comma-separated paths to unlabeled file list",
    )
    parser.add_argument("--task", help=f"The currently supported options: {SUPPORTED_TASKS}.", default="")
    parser.add_argument(
        "--train-type",
        help=f"The currently supported options: {TASK_TYPE_TO_SUB_DIR_NAME.keys()}.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--workspace",
        help="Path to the workspace where the command will run.",
        default=None,
    )
    parser.add_argument(
        "--model", help="Enter the name of the model you want to use. (Ex. EfficientNet-B0).", default=""
    )
    parser.add_argument(
        "--backbone",
        help="Available Backbone Type can be found using 'otx find --backbone {framework}'.\n"
        "If there is an already created backbone configuration yaml file, enter the corresponding path.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Set deterministic to True, default=False.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Set seed for configuration.",
    )

    return parser.parse_args()


def main():
    """Main function for model or backbone or task building."""

    args = get_args()
    config_manager = ConfigManager(args, workspace_root=args.workspace, mode="build")
    if args.task:
        config_manager.task_type = args.task.upper()

    # Auto-Configuration for model template
    config_manager.configure_template(model=args.model)

    config_manager.build_workspace(new_workspace_path=args.workspace)

    # Auto-Configuration for Dataset configuration
    config_manager.configure_data_config()

    # Build Backbone related
    if args.backbone:
        from otx.cli.builder import Builder

        builder = Builder()
        missing_args = []
        if not args.backbone.endswith((".yml", ".yaml", ".json")):
            backbone_config_file = str(config_manager.workspace_root / "backbone.yaml")
            missing_args = builder.build_backbone_config(args.backbone, backbone_config_file)
        else:
            backbone_config_file = args.backbone
        if missing_args:
            print(
                f"[!] {args.backbone} backbone has inputs that the user must enter.\n"
                f"[!] Edit {backbone_config_file} and run 'otx build --backbone backbone.yaml'."
            )
        else:
            builder.merge_backbone(config_manager.workspace_root / "model.py", backbone_config_file)

    return dict(retcode=0, task_type=args.task)


if __name__ == "__main__":
    main()
