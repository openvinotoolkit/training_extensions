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
from typing import List, Tuple, Union


from otx.cli.utils.experiment import set_arguments_to_cmd
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


def _get_default_workspace_path(template: Union[str, Path]) -> Path:
    """Provide default workspace based on model template.

    Args:
        template (Union[str, Path]): model template path

    Returns:
        Path: workspace path
    """
    with open(template, "r") as f:
        template_file = yaml.safe_load(f)
    model_name = template_file["name"]
    workspace_path = Path(f"./{model_name}_{uuid.uuid4().hex}")

    return workspace_path


def _get_cmd_list(what_to_run: List[str]) -> List[Tuple[str, List[str]]]:
    """Parse arguments for otx and return list of pair of otx entry and arguments.

    Args:
        what_to_run (List[str]): OTX arguments to run.

    Returns:
        List[Tuple[str, List[str]]]: list of tuples each of which contains otx cli entry and arguments for it.
    """
    otx_cli_entry = [val.split('_')[1] for val in __all__]
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

    return cmd_list


def _run_commands(cmd_list: List[Tuple[str, List[str]]], template: str, workspace_path: Path):
    """Run all commands.

    Args:
        cmd_list (List[Tuple[str, List[str]]]):
            list of tuples each of which contains otx cli entry and arguments for it.
        template (str): model template path.
        workspace_path (Path): workspace path
    """

    output_model_name = {
        "export" : "openvino.bin",
        "optimize" : "weights.pth",
    }

    def find_model_path(cmd_entry):
        output_dir = list((workspace_path / "outputs").glob(f"*{cmd_entry}"))
        if not output_dir:
            print(
                f"'otx {cmd_entry}' was executed right before, but there is no output directory. "
                "Evaluating the model is skipped."
            )
            return None
        file_path = list(output_dir[0].rglob(output_model_name[cmd_entry]))
        if not file_path:
            print(
                f"'otx {cmd_entry}' was executed right before, but {output_model_name[cmd_entry]} can't be found. "
                "Evaluating the model is skipped."
            )
            return None
        return file_path[0]
    
    previous_cmd = None
    for cmd, cmd_args in cmd_list:
        if cmd == "eval":
            if previous_cmd in output_model_name:
                file_path = find_model_path(previous_cmd)
                if file_path is None:
                    continue
                set_arguments_to_cmd(cmd_args, "--load-weights", str(file_path))
                output_path = str(file_path.parents[1])
            else:
                output_path = str(workspace_path / "outputs" / "latest_trained_model")
            set_arguments_to_cmd(cmd_args, "--output", output_path)

        set_arguments_to_cmd(cmd_args, "--workspace", str(workspace_path))

        sys.argv = [f"otx {cmd}", template] + cmd_args
        globals()[f"otx_{cmd}"]()

        previous_cmd = cmd


def main():
    args, what_to_run = parse_args()
    template = args.template

    if args.workspace is None:
        workspace_path = _get_default_workspace_path(template)
    else:
        workspace_path = Path(args.workspace)

    cmd_list = _get_cmd_list(what_to_run)

    _run_commands(cmd_list, template, workspace_path)

    return dict(retcode=0)


if __name__ == "__main__":
    main()
