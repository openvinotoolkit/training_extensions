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
import csv
import os
import sys
import json
import statistics
import shutil
import yaml
from math import sqrt
from copy import copy
from datetime import datetime
from pathlib import Path
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


def parse_performance(performance_file: Path, with_fps: bool = False):
    with performance_file.open("r") as f:
        temp = json.load(f)
    if with_fps:
        return temp["f-measure"], temp["avg_time_per_image"]
    return temp["f-measure"]


def organize_exp_result(workspace: Union[str, Path], exp_meta: Dict[str, str]):
    if isinstance(workspace, str):
        workspace = Path(workspace)

    exp_result = {
        "val_score" : 0,
        "test_score" : None,
        "export_model_score" : None,
        "iter_time_arr" : [],
        "data_time_arr" : [],
        "export_model_speed" : None,
        "max_cpu_mem(GiB)" : None,
        "avg_cpu_util(%)" : None,
        "max_gpu_mem(GiB)" : None,
        "avg_gpu_util(%)" : None,
    }
    for task_dir in (workspace / "outputs").iterdir():
        if "train" in str(task_dir.name):
            # test score
            performance_file = list(task_dir.glob("performance.json"))
            if performance_file:
                exp_result["test_score"] = parse_performance(performance_file[0])

            # best eval score & iter, data time
            train_history_file = list((task_dir / "logs").glob("*.log.json"))[0]
            with train_history_file.open("r") as f:
                lines = f.readlines()

            for line in lines:
                iter_history = json.loads(line)
                if iter_history.get("mode") == "train":
                    exp_result["iter_time_arr"].append(iter_history["time"])
                    exp_result["data_time_arr"].append(iter_history["data_time"])
                elif iter_history.get("mode") == "val":
                    if exp_result["val_score"] < iter_history["mAP"]:
                        exp_result["val_score"] = iter_history["mAP"]

            resource_file = task_dir / "resource_usage.yaml"
            if resource_file.exists():
                with resource_file.open("r") as f:
                    resource_usage = yaml.safe_load(f)
            
                for key in ["max_cpu_mem(GiB)", "avg_cpu_util(%)", "max_gpu_mem(GiB)" ,"avg_gpu_util(%)"]:
                    exp_result[key] = resource_usage[key]

        elif "export" in str(task_dir):
            performance_file = list(task_dir.glob("performance.json"))
            if performance_file:
                exp_result["export_model_score"], exp_result["export_model_speed"] \
                    = parse_performance(performance_file[0], True)

    for iter_type in ["iter_time", "data_time"]:
        if exp_result[f"{iter_type}_arr"]:
            exp_result[f"avg_{iter_type}"] = round(statistics.mean(exp_result[f"{iter_type}_arr"]), 4)
            exp_result[f"std_{iter_type}"] = round(statistics.stdev(exp_result[f"{iter_type}_arr"]), 4)
        else:
            exp_result[f"avg_{iter_type}"] = None
            exp_result[f"std_{iter_type}"] = None

        del exp_result[f"{iter_type}_arr"]

    with (workspace / "exp_result.yaml").open("w") as f:
        yaml.dump(
            {
                "meta" : exp_meta,
                "exp_result" : exp_result,
            },
            f,
            default_flow_style=False
        )
    
    
def aggregate_all_exp_result(exp_dir: Union[str, Path]):
    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)


    tensorboard_dir = exp_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    exp_result_summary = {}
    all_exp_result = []
    header = [
        "val_score",
        "test_score",
        "export_model_score",
        "export_model_speed",
        "avg_iter_time",
        "std_iter_time",
        "avg_data_time",
        "std_data_time",
        "max_cpu_mem(GiB)",
        "avg_cpu_util(%)",
        "max_gpu_mem(GiB)",
        "avg_gpu_util(%)",
    ]
    stdev_field = ["std_iter_time", "std_data_time"]
    add_meta_to_header = False

    for each_exp in exp_dir.iterdir():
        if not (each_exp / "exp_result.yaml").exists():
            continue

        with (each_exp / "exp_result.yaml").open("r") as f:
            exp_result: Dict[str, Dict] = yaml.safe_load(f)

        if not add_meta_to_header:
            header = list(exp_result["meta"].keys()) + header
            add_meta_to_header = True

        all_exp_result.append(copy(exp_result["meta"]))
        all_exp_result[-1].update(exp_result["exp_result"])
        del exp_result["meta"]["repeat"]
        exp_name = " ".join([val for val in exp_result["meta"].values()])
        if exp_name in exp_result_summary:
            for key, val in exp_result["exp_result"].items():
                if val is None:
                    continue

                if key in stdev_field:  # avg. std is calculated by averaging variance and applying sqrt later
                    exp_result_summary[exp_name]["result"][key] += val ** 2
                else:
                    exp_result_summary[exp_name]["result"][key] += val
            exp_result_summary[exp_name]["num"] += 1
        else:
            exp_result_summary[exp_name] = {
                "result" : exp_result["exp_result"],
                "num" : 1,
                "meta" : exp_result["meta"]
            }
            for field in stdev_field:
                if exp_result_summary[exp_name]["result"][field] is not None:
                    exp_result_summary[exp_name]["result"][field] = exp_result_summary[exp_name]["result"][field] ** 2

        # copy tensorboard log into tensorboard dir
        exp_tb_dir = list(each_exp.rglob("tf_logs"))
        if exp_tb_dir:
            temp = tensorboard_dir / each_exp.name
            shutil.copytree(exp_tb_dir[0], temp, dirs_exist_ok=True)

    with (exp_dir / "all_exp_result.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader() 
        writer.writerows(all_exp_result)

    # process for experiment summary
    header.remove("repeat")
    for each_exp_result_summary in exp_result_summary.values():
        for key in each_exp_result_summary["result"].keys():
            if key in stdev_field:
                each_exp_result_summary["result"][key] = sqrt(
                    each_exp_result_summary["result"][key] / each_exp_result_summary["num"])
            else:
                each_exp_result_summary["result"][key] /= each_exp_result_summary["num"]

        each_exp_result_summary["result"].update(each_exp_result_summary["meta"])

    with (exp_dir / "exp_summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader() 
        writer.writerows([each_exp_result_summary["result"] for each_exp_result_summary in exp_result_summary.values()])

    # print experiment summary to console
    table = Table(title="Experiment Summary")
    for field in header:
        table.add_column(field, justify="center", no_wrap=True)
    for each_exp_result_summary in exp_result_summary.values():
        table_row = []
        for field in header:
            val = each_exp_result_summary["result"][field]
            if isinstance(val, float):
                val = round(val, 4)
            table_row.append(str(val))

        table.add_row(*table_row)

    console = Console()
    console.print(table)


def replace_var_in_str(
    variable: Dict[str, Union[str, List[str]]],
    target: str,
    keep_variable: bool = False,
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

        if keep_variable:
            ret.append({
                "variable" : {key_found[i] : val for i, val in enumerate(value_of_key_found)},
                "command" : replaced_target
            })
        else:
            ret.append(replaced_target)

    if not keep_variable and len(ret) == 1:
        return ret[0]
    return ret


def map_variable(
    variable: Dict[str, Union[str, List[str]]],
    target_dict: Dict[str, Union[str, List[str]]],
    target_key: str,
    keep_variable: bool = False,
):
    target = target_dict[target_key]
    if isinstance(target, list):
        new_arr = []
        for each_str in target:
            str_replaced = replace_var_in_str(variable, each_str, keep_variable)
            if isinstance(str_replaced, str):
                new_arr.append(str_replaced)
            else:
                new_arr.extend(str_replaced)
            
        target_dict[target_key] = new_arr
    elif isinstance(target, str):
        target_dict[target_key] = replace_var_in_str(variable, target, keep_variable)


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

    # avg_exp_result = {}
    current_dir = os.getcwd()
    os.chdir(output_path)
    for command_info in command_list:
        original_command = command_info["command"]
        command_var = command_info["variable"]
        exp_name = "_".join(command_var.values())

        for repeat_idx in range(repeat):
            workspace = exp_name.replace('/', '_') + f"_repeat{repeat_idx}"
            command_var["repeat"] = str(repeat_idx)

            command = copy(original_command).split()
            command = command[:3] + ["--workspace", workspace] + command[3:]

            if "train" in command:
                train_idx = command.index("train")
                command = command[:train_idx+1] + ["--seed", str(repeat_idx)] + command[train_idx+1:]

            sys.argv = [" ".join(command[:2])] + command[2:]
            globals()["_".join(sys.argv[0].split())]()

            workspace = Path(workspace)
            organize_exp_result(workspace, command_var)
    os.chdir(current_dir)

    aggregate_all_exp_result(output_path)


def main():
    exp_recipe = get_exp_recipe()
    run_experiment_recipe(exp_recipe)

    return dict(retcode=0)


if __name__ == "__main__":
    main()
