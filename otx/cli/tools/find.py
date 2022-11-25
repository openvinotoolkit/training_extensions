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
import os

from prettytable import PrettyTable

from otx.cli.registry import Registry


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="A root dir where templates should be searched.", default="otx")
    parser.add_argument("--task_type")
    parser.add_argument("--template", action="store_true")
    parser.add_argument("--backbone", action="append")
    parser.add_argument("--save-to", help="")

    return parser.parse_args()


def main():
    """Main function for model templates searching."""

    args = parse_args()

    otx_registry = Registry(args.root)
    if args.task_type:
        otx_registry = otx_registry.filter(task_type=args.task_type)

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
        backbone_registry_lst = otx_registry.find_backbones(args.backbone)
        row_index = 0
        backbone_table = PrettyTable(["index", "Backbone Type"])
        for backend, backbone_lst in backbone_registry_lst.items():
            for backbone in backbone_lst:
                scope_name = "mmdet" if backend == "pytorchcv" else backend
                backbone_table.add_row([row_index + 1, f"{scope_name}.{backbone}"])
                row_index += 1
        print(backbone_table)


if __name__ == "__main__":
    main()
