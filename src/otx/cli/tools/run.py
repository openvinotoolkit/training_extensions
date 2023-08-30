"""OTX CLI entry point."""

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
import sys
import uuid
import yaml
from pathlib import Path
from typing import List, Tuple


from .build import main as otx_build
from .demo import main as otx_demo
from .deploy import main as otx_deploy
from .eval import main as otx_eval
from .explain import main as otx_explain
from .export import main as otx_export
from .find import main as otx_find
from .optimize import main as otx_optimize
from .train import main as otx_train

__all__ = [
    "otx_demo",
    "otx_deploy",
    "otx_eval",
    "otx_explain",
    "otx_export",
    "otx_find",
    "otx_train",
    "otx_optimize",
    "otx_build",
]


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("template", nargs="?")
    parser.add_argument(
        "--workspace",
        help="Location where the intermediate output of the training will be stored.",
    )

    return parser.parse_known_args()


def main():
    """Entry point for OTX CLI.

    This function is a single entry point for all OTX CLI related operations:
      - build
      - demo
      - deploy
      - eval
      - explain
      - export
      - find
      - train
      - optimize
    """

    otx_cli_entry = [val.split('_')[1] for val in __all__]
    args, what_to_run = parse_args()
    template = args.template

    if args.workspace is None:
        with open(template, "r") as f:
            template_file = yaml.safe_load(f)
        model_name = template_file["name"]
        workspace_path = Path(f"./{model_name}_{uuid.uuid4().hex}")
    else:
        workspace_path = Path(args.workspace)

    cmd_list: List[Tuple[str, List[str]]] = []
    cmd = None
    for arg in what_to_run:
        if arg in otx_cli_entry:
            if cmd is not None:
                cmd_list.append((cmd, arg_for_cmd))
            cmd = arg
            arg_for_cmd = []
        else:
            arg_for_cmd.append(arg)
    if cmd is not None:
        cmd_list.append((cmd, arg_for_cmd))

    previous_cmd = None
    for cmd, cmd_args in cmd_list:
        if cmd == "eval":
            output_path = str(workspace_path / "outputs" / "latest_trained_model")
            if previous_cmd == "export":
                ov_path = list(workspace_path.rglob("openvino.bin"))
                if ov_path:
                    ov_path = ov_path[0]
                    cmd_args = ["--load-weights", str(ov_path)] + cmd_args
                    output_path = str(ov_path.parents[1])
            cmd_args = ["--output", output_path] + cmd_args

        if "--load-weights" not in cmd_args:
            cmd_args = ["--workspace", str(workspace_path)] + cmd_args

        cmd_args.insert(0, template)
        sys.argv = [f"otx {cmd}"] + cmd_args
        previous_cmd = cmd

        print("*"*100, sys.argv)
        globals()[f"otx_{cmd}"]()

    return dict(retcode=0)


if __name__ == "__main__":
    main()
