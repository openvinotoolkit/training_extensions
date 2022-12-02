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
from otx.cli.utils.importing import get_backbone_registry, get_required_args

# pylint: disable=too-many-locals

SUPPORTED_TASKS = (
    "CLASSIFICATION",
    "DETECTION",
    "INSTANCE_SEGMENTATION",
    "SEGMENTATION",
    "ACTION_CLASSIFICATION",
    "ANOMALY_CLASSIFICATION",
    "ANOMALY_DETECTION",
    "ANOMALY_SEGMENTATION",
)

SUPPORTED_BACKBONE_BACKENDS = (
    "otx",
    "mmcls",
    "mmdet",
    "mmseg",
    "torchvision",
    "pytorchcv",
)


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="A root dir where templates should be searched.", default="otx")
    parser.add_argument("--task", help=f"The currently supported options: {SUPPORTED_TASKS}.")
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

    otx_registry = Registry(args.root)
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

    if args.backbone:
        backbone_registry = {}
        all_backbone_lst = otx_registry.get_backbones(args.backbone)
        for backend in args.backbone:
            registry, _ = get_backbone_registry(backend)
            backbone_registry[backend] = registry
        row_index = 0
        backbone_table = PrettyTable(["Index", "Backbone Type", "Required Args"])
        for backend, backbone_lst in all_backbone_lst.items():
            for backbone in backbone_lst:
                scope_name = "mmdet" if backend == "pytorchcv" else backend
                backbone_type = f"{scope_name}.{backbone}"
                required_arg = get_required_args(backbone_registry[backend].get(backbone_type))
                required_arg = ", ".join(required_arg) if required_arg else ""
                backbone_table.add_row([row_index + 1, backbone_type, required_arg])
                row_index += 1
        print(backbone_table)


if __name__ == "__main__":
    main()
