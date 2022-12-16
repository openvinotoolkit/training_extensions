"""OTX searching command 'otx find'.

Through this command, you can check the tasks, templates, and backbones available in OTX.
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

from prettytable import PrettyTable

from otx.cli.registry import Registry
from otx.cli.utils.importing import SUPPORTED_BACKBONE_BACKENDS, get_otx_root_path

# pylint: disable=too-many-locals

SUPPORTED_TASKS = (
    "CLASSIFICATION",
    "DETECTION",
    "ROTATED_DETECTION",
    "INSTANCE_SEGMENTATION",
    "SEGMENTATION",
    "ACTION_CLASSIFICATION",
    "ANOMALY_CLASSIFICATION",
    "ANOMALY_DETECTION",
    "ANOMALY_SEGMENTATION",
)


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help=f"Supported task types.", choices=SUPPORTED_TASKS)
    parser.add_argument(
        "--template", action="store_true", help="Shows a list of templates that can be used immediately."
    )
    parser.add_argument(
        "--backbone",
        action="append",
        help=f"The currently supported options: {SUPPORTED_BACKBONE_BACKENDS}.",
    )

    return parser.parse_args()


def main():
    """Main function for model templates searching."""

    args = parse_args()

    otx_root = get_otx_root_path()
    otx_registry = Registry(otx_root)
    if args.task:
        otx_registry = otx_registry.filter(task_type=args.task)

    if args.template:
        template_table = PrettyTable(["TASK", "ID", "NAME", "PATH"])
        for template in otx_registry.templates:
            relpath = os.path.relpath(template.model_template_path, os.path.abspath("."))
            template_table.add_row(
                [
                    template.task_type,
                    template.model_template_id,
                    template.name,
                    relpath,
                ]
            )
        print(template_table)

    # TODO: Get params from cli args & Flow arrangement (for all tasks backbone usable)
    if args.backbone:
        all_backbones = otx_registry.get_backbones(args.backbone)
        backbone_table = PrettyTable(["Index", "Backbone Type", "Required Args", "Confirmed model"])
        row_index = 0
        for _, backbone_meta in all_backbones.items():
            for backbone_type, meta_data in backbone_meta.items():
                required_args = []
                for arg in meta_data["required"]:
                    output_arg = f"{arg}"
                    if arg in meta_data["options"]:
                        output_arg += f"={meta_data['options'][arg]}"
                    required_args.append(output_arg)
                row = [
                    row_index + 1,
                    backbone_type,
                    ", ".join(required_args) if required_args else "",
                    ", ".join(meta_data["available"]) if meta_data["available"] else "",
                ]
                backbone_table.add_row(row)
                row_index += 1
        print(backbone_table)


if __name__ == "__main__":
    main()
