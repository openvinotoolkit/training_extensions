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
import os
import sys
import json
import statistics
import shutil
from copy import copy
from datetime import datetime
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


def parse_performance(output_dir: Path, with_fps: bool = False):
    performance_file = output_dir / "performance.json"
    if not performance_file.exists():
        raise RuntimeError(f"{performance_file} doesn't exist.")

    with performance_file.open("r") as f:
        temp = json.load(f)
    if with_fps:
        return temp["f-measure"], temp["avg_time_per_image"]
    return temp["f-measure"]


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
    exported_model_fps = None
    for task_dir in (workspace / "outputs").iterdir():
        if "train" in str(task_dir.name):
            # test score
            test_score = parse_performance(task_dir)

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
            
                max_cpu_mem = " ".join(lines[0].split()[1:])
                avg_cpu_util = " ".join(lines[1].split()[1:])
                max_gpu_mem = " ".join(lines[2].split()[1:])
                avg_gpu_util = " ".join(lines[3].split()[1:])

        elif "export" in str(task_dir):
            export_model_score, exported_model_fps = parse_performance(task_dir, True)

    with (workspace / "exp_result.txt").open("w") as f:
        f.write(
            f"best_eval_score\t{val_score}\n"
            f"test_score\t{test_score}\n"
            f"export_score\t{round(export_model_score, 4)}\n"
            f"export_infer_speed\t{exported_model_fps}\n"
            f"avg_iter_time\t{round(statistics.mean(iter_time_arr), 4)}\n"
            f"std_iter_time\t{round(statistics.stdev(iter_time_arr), 4)}\n"
            f"avg_data_time\t{round(statistics.mean(data_time_arr), 4)}\n"
            f"std_data_time\t{round(statistics.stdev(data_time_arr), 4)}\n"
            f"max_cpu_mem\t{max_cpu_mem}\n"
            f"avg_cpu_util\t{avg_cpu_util}\n"
            f"max_gpu_mem\t{max_gpu_mem}\n"
            f"avg_gpu_util\t{avg_gpu_util}\n"
        )
    
    
def aggregate_all_exp_result(exp_dir: Union[str, Path], avg_exp_result: Dict[str, Dict[str, List]]):
    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)

    table = Table(title="Experiment Summary")

    output_file = (exp_dir / "exp_table.txt").open("w")
    output_file.write("experiment detail\n")
    write_headers = False
    headers = []

    tensorboard_dir = exp_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    exp_dir_arr = []
    for exp_history in avg_exp_result.values():
        exp_dir_arr.extend(exp_history["exp_name"])

    # for each_exp in exp_dir.iterdir():
    for each_exp in exp_dir_arr:
        each_exp = exp_dir / each_exp
        with (each_exp / "exp_info.json").open("r") as f:
            exp_info = json.load(f)
        exp_result = each_exp / "exp_result.txt"
        if exp_result.exists():
            with exp_result.open("r") as f:
                lines = f.readlines()
            # write a head
            if not write_headers:
                for key in exp_info.keys():
                    headers.append(key)
                for line in lines:
                    header = line.split()[0]
                    headers.append(header)
                output_file.write("\t".join(headers) + '\n')
                write_headers = True

            # write a experiment result
            row = [val for val in exp_info.values()]
            for line in lines:
                row.append(line.split()[1])
            output_file.write("\t".join(row))
            output_file.write('\n')

            # sum all repeated experiment results
            for exp_info in avg_exp_result.values():
                if each_exp.name in exp_info["exp_name"]:
                    for idx, val in enumerate(row[len(exp_info):]):
                        val = float(val)
                        if len(exp_info["value"]) <= idx:
                            exp_info["value"].append(val)
                        else:
                            exp_info["value"][idx] += val
                    break

        # copy tensorboard log into tensorboard dir
        exp_tb_dir = list(each_exp.rglob("tf_logs"))
        if exp_tb_dir:
            temp = tensorboard_dir / each_exp.name
            shutil.copytree(exp_tb_dir[0], temp, dirs_exist_ok=True)

    headers.remove("repeat")
    output_file.write("\nexperiment summary\n")
    for header in headers:
        table.add_column(header, justify="center")
        output_file.write(f"{header}\t")
    output_file.write("\n")
    for exp_info in avg_exp_result.values():
        variable = exp_info["variable"]
        output_file.write('\t'.join(variable.values()) + "\t")
        num_exp = len(exp_info["exp_name"])
        for i in range(len(exp_info["value"])):
            exp_info["value"][i] = round(exp_info["value"][i] / num_exp, 4)
            output_file.write(f"{exp_info['value'][i]}\t")
        output_file.write('\n')
        table.add_row(*[val for val in variable.values()], *[str(val) for val in exp_info["value"]])

    output_file.close()
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

    ret = []
    values_of_key_found = []
    for key in key_found:
        if isinstance(variable[key], list):
            values_of_key_found.append(variable[key])
        else:
            values_of_key_found.append([variable[key]])

    for value_of_key_found in product(*values_of_key_found):
        replaced_target = copy(target)
        for key, val in zip(key_found, value_of_key_found):
            replaced_target = replaced_target.replace(f"${{{key}}}", val)

        if keep_key:
            ret.append({
                "variable" : {key_found[i] : val for i, val in enumerate(value_of_key_found)},
                "command" : replaced_target
            })
        else:
            ret.append(replaced_target)

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

    avg_exp_result = {}
    current_dir = os.getcwd()
    os.chdir(output_path)
    for command_info in command_list:
        original_command = command_info["command"]
        command_var = command_info["variable"]
        exp_name = "_".join(command_var.values())

        for repeat_idx in range(repeat):
            workspace = exp_name.replace('/', '_') + f"_repeat{repeat_idx}"
            if exp_name not in avg_exp_result:
                avg_exp_result[exp_name] = {"exp_name" : [], "value" : [], "variable" : copy(command_var)}
            avg_exp_result[exp_name]["exp_name"].append(workspace)
            command_var["repeat"] = str(repeat_idx)

            command = copy(original_command).split()
            command = command[:3] + ["--workspace", workspace] + command[3:]

            if "train" in command:
                train_idx = command.index("train")
                command = command[:train_idx+1] + ["--seed", str(repeat_idx)] + command[train_idx+1:]

            sys.argv = [" ".join(command[:2])] + command[2:]
            globals()["_".join(sys.argv[0].split())]()

            workspace = Path(workspace)
            with (workspace / "exp_info.json").open("w") as f:
                json.dump(command_var, f)

            organize_exp_result(workspace)
    os.chdir(current_dir)

    aggregate_all_exp_result(output_path, avg_exp_result)


def main():
    exp_recipe = get_exp_recipe()
    run_experiment_recipe(exp_recipe)

    return dict(retcode=0)


if __name__ == "__main__":
    main()
