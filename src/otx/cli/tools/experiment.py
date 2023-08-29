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
import yaml
import re
from datetime import datetime
import os
import sys
import json
import statistics
from pathlib import Path
from collections import OrderedDict
from itertools import product
from typing import Union, Dict, List

from rich.console import Console
from rich.table import Table

from .build import main as otx_build
from .demo import main as otx_demo
from .deploy import main as otx_deploy
from .eval import main as otx_eval
from .explain import main as otx_explain
from .export import main as otx_export
from .find import main as otx_find
from .optimize import main as otx_optimize
from .train import main as otx_train
from .run import main as otx_run


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
    "otx_run",
]

def get_args() -> str:
    """Parses command line arguments."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", "--file", type=str, required=True)
    return parser.parse_args()


def get_exp_recipe() -> Dict:
    args = get_args() 
    file_path = args.file

    if not os.path.exists(file_path):
        raise RuntimeError(f"{file_path} doesn't exist.")

    with open(file_path, "r") as f:
        exp_recipe = yaml.safe_load(f)

    return exp_recipe


def parse_eval_score(output_dir: Path):
    performance_file = output_dir / "performance.json"
    if performance_file.exists():
        with performance_file.open("r") as f:
            temp = json.load(f)
        return temp["f-measure"]
    return 0


def organize_exp_result(workspace: Union[str, Path]):
    if isinstance(workspace, str):
        workspace = Path(workspace)

    test_score = None
    export_model_score = None
    iter_time_arr = []
    data_time_arr = []
    val_score = 0
    resource_file = None
    max_cpu_mem = None
    max_gpu_mem = None
    avg_gpu_util = None
    for task_dir in (workspace / "outputs").iterdir():
        if "train" in str(task_dir.name):
            # test score
            test_score = parse_eval_score(task_dir)

            # best eval score & iter, data time
            train_history_file = list((task_dir / "logs").glob("*.log.json"))[0]
            with train_history_file.open("r") as f:
                lines = f.readlines()

            for line in lines:
                each_info = json.loads(line)
                if each_info.get("mode") == "train":
                    iter_time_arr.append(each_info["time"])
                    data_time_arr.append(each_info["data_time"])
                elif each_info.get("mode") == "val":
                    if val_score < each_info["mAP"]:
                        val_score = each_info["mAP"]

            if (task_dir / "resource.txt").exists():
                resource_file = task_dir / "resource.txt"
                with resource_file.open("r") as f:
                    lines = f.readlines()
            
                max_cpu_mem = lines[0]
                max_gpu_mem = lines[1]
                avg_gpu_util = lines[2]

        elif "export" in str(task_dir):
            export_model_score = parse_eval_score(task_dir)

    with (workspace / "exp_result.txt").open("w") as f:
        f.write(
            f"best_eval_score\t{val_score}\n"
            f"test_score\t{test_score}\n"
            f"export_score\t{export_model_score}\n"
            f"avg_iter_time\t{statistics.mean(iter_time_arr)}\n"
            f"var_iter_time\t{statistics.variance(iter_time_arr)}\n"
            f"avg_data_time\t{statistics.mean(data_time_arr)}\n"
            f"var_data_time\t{statistics.variance(data_time_arr)}\n"
            f"{max_cpu_mem}"
            f"{max_gpu_mem}"
            f"{avg_gpu_util}"
        )
    
    
def aggregate_exp_result(exp_dir: Union[str, Path]):
    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)

    table = Table(title="Experiment Summary")

    output_file = (exp_dir / "exp_table.txt").open("w")
    output_file.write("name\t")
    table.add_column("name", justify="center")
    write_type = False

    for each_exp in exp_dir.iterdir():
        exp_result = each_exp / "exp_result.txt"
        if exp_result.exists():
            with exp_result.open("r") as f:
                lines = f.readlines()
                if not write_type:
                    for line in lines:
                        header = line.split()[0]
                        table.add_column(header, justify="center")
                        output_file.write(header + '\t')
                    output_file.write('\n')
                    write_type = True

                row = [each_exp.name]
                for line in lines:
                    row.append(line.split()[1])
                output_file.write("\t".join(row))
                output_file.write('\n')
                table.add_row(*row)

    console = Console()
    console.print(table)


def replace_var_in_str(
    variable: Dict[str, Union[str, List[str]]],
    target: str,
    keep_key: bool = False,
) -> Union[str, List[str], Dict[str, str]]:
    replace_pat = re.compile(r"\$\{(\w+)\}")
    key_found = [x for x in set(replace_pat.findall(target)) if x in variable]
    if not key_found:
        return target

    ret = OrderedDict() if keep_key else []
    values_of_key_found = []
    for key in key_found:
        if isinstance(variable[key], list):
            values_of_key_found.append(variable[key])
        else:
            values_of_key_found.append([variable[key]])

    for value_of_key_found in product(*values_of_key_found):
        for key, val in zip(key_found, value_of_key_found):
            target = target.replace(f"${{{key}}}", val)

        if keep_key:
            ret["_".join(value_of_key_found)] = target
        else:
            ret.append(target)

    if not keep_key and len(ret) == 1:
        return ret[0]
    return ret


def map_variable(
    variable: Dict[str, Union[str, List[str]]],
    target_dict: Dict[str, Union[str, List[str]]],
    target_key: str,
    keep_key: bool = False,
):
    target = target_dict[target_key]
    if isinstance(target, list):
        new_arr = []
        for each_str in target:
            str_replaced = replace_var_in_str(variable, each_str, keep_key)
            if isinstance(str_replaced, str):
                new_arr.append(str_replaced)
            else:
                new_arr.extend(str_replaced)
            
        target_dict[target_key] = new_arr
    elif isinstance(target, str):
        target_dict[target_key] = replace_var_in_str(variable, target, keep_key)


def get_command_list(exp_recipe: Dict) -> Dict[str, str]:
    constants: Dict = exp_recipe.get("constants", {})
    variables: Dict = exp_recipe.get("variables", {})

    for key in variables.keys():
        map_variable(constants, variables, key)
    map_variable(constants, exp_recipe, "command")
    map_variable(variables, exp_recipe, "command", True)

    return exp_recipe["command"]


def run_experiment_recipe(exp_recipe: Dict):
    output_path = Path(exp_recipe.get("output_path", f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    output_path.mkdir(exist_ok=True)
    repeat = exp_recipe.get("repeat", 1)

    command_list = get_command_list(exp_recipe)

    current_dir = os.getcwd()
    os.chdir(output_path)
    for exp_name, command in command_list.items():
        exp_name = exp_name.replace('/', '_')

        command_split = command.split()
        command_split.insert(2, f"--workspace {exp_name}")
        command = " ".join(command_split)

        sys.argv = [" ".join(command.split()[:2])] + command.split()[2:]
        for _ in range(repeat):
            globals()["_".join(sys.argv[0].split())]()
    os.chdir(current_dir)

    for exp_dir in output_path.iterdir():
        organize_exp_result(exp_dir)

    aggregate_exp_result(output_path)


def main():
    exp_recipe = get_exp_recipe()
    run_experiment_recipe(exp_recipe)

    return dict(retcode=0)


if __name__ == "__main__":
    main()
