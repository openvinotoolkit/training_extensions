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
from textwrap import fill

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
    "ACTION_DETECTION",
    "VISUAL_PROMPTING",
    "ANOMALY_CLASSIFICATION",
    "ANOMALY_DETECTION",
    "ANOMALY_SEGMENTATION",
)


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help=f"The currently supported options: {SUPPORTED_TASKS}.")
    parser.add_argument(
        "--template", action="store_true", help="Shows a list of templates that can be used immediately."
    )
    parser.add_argument(
        "--backbone",
        nargs="+",
        help=f"The currently supported options: {list(SUPPORTED_BACKBONE_BACKENDS.keys())}.",
    )

    return parser.parse_args()


def generate_backbone_rows(index: int, backbone_type: str, meta_data: dict):
    """Generate table row for backbone json format.

    It expects a json file format from otx/cli/builder/supported_backbone.
    index: The index of each backbone (int)
    backbone_type: The backbone type want to add (str)
    meta_data: This is the metadata of the backbone type (dict)
        Metadata keys expect required, options, and available.
    """
    max_row_width = 40
    rows = []
    required_args = meta_data["required"] if meta_data["required"] else [""]
    required_options = meta_data["options"]
    use_backbone_type = False
    for arg in required_args:
        row = []
        options = list(map(str, required_options[arg])) if arg in required_options else []
        if use_backbone_type:
            row.append("")  # Index
            row.append("")  # backbone_type
        else:
            row.append(str(index))  # Index
            row.append(backbone_type)  # backbone_type
            use_backbone_type = True
        row.append(arg)  # Required-Args
        option_str = ", ".join(options) if options else ""
        row.append(fill(option_str, width=max_row_width))  # Options
        rows.append(row)
    return rows


def main():
    """Main function for model templates & backbone searching.

    When the template argment is input, the templates based on the otx folder are displayed.
    Given a backbone argument as input,
    it displays a list of backbones available in the backend of the relevant task.
    """

    args = parse_args()

    otx_root = get_otx_root_path()
    otx_registry = Registry(otx_root).filter(task_type=args.task)

    if not args.backbone or args.template:
        template_table = PrettyTable(["TASK", "ID", "NAME", "BASE PATH"])
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

    if args.backbone:
        all_backbones = otx_registry.get_backbones(args.backbone)
        backbone_table = PrettyTable(["Index", "Backbone Type", "Required-Args", "Options"])
        row_index = 1
        for _, backbone_meta in all_backbones.items():
            for backbone_type, meta_data in backbone_meta.items():
                available_task = meta_data.get("available", [])
                if not available_task or (args.task and args.task.upper() not in available_task):
                    continue
                rows = generate_backbone_rows(row_index, backbone_type, meta_data)
                backbone_table.add_rows(rows)
                row_index += 1
        print(backbone_table)

    return dict(retcode=0, task_type=args.task)


if __name__ == "__main__":
    main()
