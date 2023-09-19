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
import dataclasses
from abc import ABC, abstractmethod
from math import sqrt
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product
from typing import Union, Dict, List, Any

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


@dataclass
class ExperimentResult:
    val_score: Union[float, None] = None
    test_score: Union[float, None] = None
    train_e2e_time: Union[timedelta, None] = None
    avg_iter_time: Union[float, None] = None
    std_iter_time: Union[float, None] = None
    avg_data_time: Union[float, None] = None
    std_data_time: Union[float, None] = None
    export_model_score: Union[float, None] = None
    export_model_speed: Union[float, None] = None
    max_cpu_mem: Union[float, None] = None
    avg_cpu_util: Union[float, None] = None
    max_gpu_mem: Union[float, None] = None
    avg_gpu_util: Union[float, None] = None

    def get_formatted_result(self):
        result = dataclasses.asdict(self)

        for attr_name in ["max_cpu_mem", "max_gpu_mem"]:
            max_mem = result.pop(attr_name)
            result[f"{attr_name}(GiB)"] = max_mem

        for attr_name in ["avg_cpu_util", "avg_gpu_util"]:
            res_util = result.pop(attr_name)
            result[f"{attr_name}(%)"] = res_util

        # delete None value
        for key in list(result.keys()):
            if result[key] is None:
                del result[key]
            elif isinstance(result[key], float):
                result[key] = round(result[key], 4)

        return result

    def __add__(self, obj: 'ExperimentResult'):
        new_obj = deepcopy(self)

        for attr in dataclasses.fields(self):
            self._add_if_not_none(new_obj, obj, attr.name)

        return new_obj

    @staticmethod
    def _add_if_not_none(dst_obj: 'ExperimentResult', src_obj : 'ExperimentResult', attr: str):
        dst_obj_val = getattr(dst_obj, attr)
        src_obj_val = getattr(src_obj, attr)
        if dst_obj_val is not None and src_obj_val is not None:
            setattr(dst_obj, attr, dst_obj_val + src_obj_val)
        else:
            setattr(dst_obj, attr, None)

    def __truediv__(self, divisor: Union[int, float]):
        new_obj = deepcopy(self)

        for attr in dataclasses.fields(self):
            self._divide_if_not_none(new_obj, attr.name, divisor)

        return new_obj

    @staticmethod
    def _divide_if_not_none(obj: 'ExperimentResult', attr: str, divisor: Union[int, float]):
        obj_val = getattr(obj, attr)
        if obj_val is not None:
            setattr(obj, attr, obj_val / divisor)

    def parse_formatted_dict(self, formatted_dict: Dict):
        max_mem_pat = re.compile(r"max_.*_mem")
        cpu_util_pat = re.compile(r"avg.*_util")
        for key, val in formatted_dict.items():
            max_mem_name = max_mem_pat.search(key)
            cpu_util_name = cpu_util_pat.search(key)
            if max_mem_name is not None:
                max_mem_name = max_mem_name.group(0)
                setattr(self, max_mem_name, val)
            elif cpu_util_name is not None:
                cpu_util_name = cpu_util_name.group(0)
                setattr(self, cpu_util_name, val)
            elif key == "train_e2e_time":
                time_delta = datetime.strptime(val, "%H:%M:%S.%f") - datetime(1900, 1, 1)
                setattr(self, key, time_delta)
            else:
                setattr(self, key, val)


class BaseExpParser(ABC):
    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._exp_result = ExperimentResult()
        self._iter_time_arr = []
        self._data_time_arr = []

    @property
    def exp_result(self) -> ExperimentResult:
        return self._exp_result

    @abstractmethod
    def parse_exp_log(self):
        raise NotImplementedError

    def get_exp_result(self):
        self._calculate_avg_std_per_iter()

        return self._exp_result.get_formatted_result()

    def _calculate_avg_std_per_iter(self):
        if self._iter_time_arr:
            self._exp_result.avg_iter_time = statistics.mean(self._iter_time_arr)
            self._exp_result.std_iter_time = statistics.stdev(self._iter_time_arr)

        if self._data_time_arr:
            self._exp_result.avg_data_time = statistics.mean(self._data_time_arr)
            self._exp_result.std_data_time = statistics.stdev(self._data_time_arr)

    def _parse_eval_output(self, file_path: Path):
        with file_path.open("r") as f:
            eval_output = json.load(f)

        if "train" in str(file_path.parent.name):
            self._exp_result.test_score = eval_output["f-measure"]
        else:  # export
            if "avg_time_per_image" in eval_output:
                self._exp_result.export_model_speed = eval_output["avg_time_per_image"]
            self._exp_result.export_model_score = eval_output["f-measure"]

        
    def _parse_resource_usage(self, file_path: Path):
        with file_path.open("r") as f:
            resource_usage = yaml.safe_load(f)
    
        self._exp_result.max_cpu_mem = resource_usage["max_cpu_mem(GiB)"]
        self._exp_result.avg_cpu_util = resource_usage["avg_cpu_util(%)"]
        self._exp_result.max_gpu_mem = resource_usage["max_gpu_mem(GiB)"]
        self._exp_result.avg_gpu_util = resource_usage["avg_gpu_util(%)"]

    def _parse_cli_report(self, file_path: Path, save_val_score=True):
        with file_path.open("r") as f:
            lines = f.readlines()

        val_score_pattern = re.compile(r"score: Performance\(score: (\d+(\.\d*)?|\.\d+)")
        e2e_time_pattern = re.compile(r"time elapsed: '(\d+:\d+:\d+(\.\d*)?)'")
        for line in lines:
            if save_val_score:
                val_score = val_score_pattern.search(line)
                if val_score is not None:
                    self._exp_result.val_score = float(val_score.group(1))

            e2e_time = e2e_time_pattern.search(line)
            if e2e_time is not None:
                self._exp_result.train_e2e_time = e2e_time.group(1)


class MMCVExpParser(BaseExpParser):
    def parse_exp_log(self):
        for task_dir in (self._workspace / "outputs").iterdir():
            if "train" in str(task_dir.name):
                # test score
                eval_files = list(task_dir.glob("performance.json"))
                if eval_files:
                    self._parse_eval_output(eval_files[0])

                # best eval score & iter, data time
                train_record_files = list((task_dir / "logs").glob("*.log.json"))
                if train_record_files:
                    self._parse_train_record(train_record_files[0])

                # train e2e time
                cli_report_files = list(task_dir.glob("cli_report.log"))
                if cli_report_files:
                    self._parse_cli_report(cli_report_files[0], False)

                # get resource info
                resource_file = task_dir / "resource_usage.yaml"
                if resource_file.exists():
                    self._parse_resource_usage(resource_file)

            elif "export" in str(task_dir):
                eval_files = list(task_dir.glob("performance.json"))
                if eval_files:
                    self._parse_eval_output(eval_files[0])

    def _parse_train_record(self, file_path : Path):
        with file_path.open("r") as f:
            lines = f.readlines()

        for line in lines:
            iter_history = json.loads(line)
            if iter_history.get("mode") == "train":
                self._iter_time_arr.append(iter_history["time"])
                self._data_time_arr.append(iter_history["data_time"])
            elif iter_history.get("mode") == "val":
                if self._exp_result.val_score is None or self._exp_result.val_score < iter_history["mAP"]:
                    self._exp_result.val_score = iter_history["mAP"]


class AnomalibExpParser(BaseExpParser):
    def parse_exp_log(self):
        for task_dir in (self._workspace / "outputs").iterdir():
            if "train" in str(task_dir.name):
                # test score
                eval_files = list(task_dir.glob("performance.json"))
                if eval_files:
                    self._parse_eval_output(eval_files[0])

                # val score and train e2e time
                cli_report_files = list(task_dir.glob("cli_report.log"))
                if cli_report_files:
                    self._parse_cli_report(cli_report_files[0])

                # get resource info
                resource_file = task_dir / "resource_usage.yaml"
                if resource_file.exists():
                    self._parse_resource_usage(resource_file)

            elif "export" in str(task_dir):
                eval_files = list(task_dir.glob("performance.json"))
                if eval_files:
                    self._parse_eval_output(eval_files[0])


def get_exp_parser(workspace: Path) -> BaseExpParser:
    with (workspace / "template.yaml").open("r") as f:
        template = yaml.safe_load(f)

    if "anomaly" in template["task_type"].lower():
        return AnomalibExpParser(workspace)
    return MMCVExpParser(workspace)


def organize_exp_result(workspace: Union[str, Path], exp_meta: Dict[str, str]):
    if isinstance(workspace, str):
        workspace = Path(workspace)

    exp_parser = get_exp_parser(workspace)
    exp_parser.parse_exp_log()

    exp_result = exp_parser.get_exp_result() 
    with (workspace / "exp_result.yaml").open("w") as f:
        yaml.dump({"meta" : exp_meta, "exp_result" : exp_result}, f, default_flow_style=False)


def write_csv(output_path: Union[str, Path], header: List[str], rows: List[Dict[str, Any]]):
    if isinstance(output_path, str):
        output_path = Path(output_path)

    with output_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader() 
        writer.writerows(rows)


def draw_rich_table(header: List[str], rows: List[Dict[str, Any]], table_title: str = "Table"):
    # print experiment summary to console
    table = Table(title=table_title)
    for field in header:
        table.add_column(field, justify="center", no_wrap=True)
    for each_exp_result_summary in rows:
        table_row = []
        for field in header:
            val = each_exp_result_summary[field]
            table_row.append(str(val))

        table.add_row(*table_row)

    console = Console()
    console.print(table)

    
def aggregate_all_exp_result(exp_dir: Union[str, Path]):
    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)

    meta_header: Union[List[str], None] = None
    metric_header = set()
    all_exp_result: List[Dict[str, str]] = []
    exp_result_aggregation = {}
    for each_exp in exp_dir.iterdir():
        exp_result_file = each_exp / "exp_result.yaml"
        if not exp_result_file.exists():
            continue

        with exp_result_file.open("r") as f:
            exp_yaml_result: Dict[str, Dict] = yaml.safe_load(f)

        each_exp_result = copy(exp_yaml_result["meta"])
        each_exp_result.update(exp_yaml_result["exp_result"])
        all_exp_result.append(each_exp_result)

        if meta_header is None:
            meta_header = list(exp_yaml_result["meta"].keys())

        metric_header = metric_header | set(exp_yaml_result["exp_result"].keys())

        exp_meta = copy(exp_yaml_result["meta"])
        exp_meta.pop("repeat")

        exp_result = ExperimentResult()
        exp_result.parse_formatted_dict(exp_yaml_result["exp_result"])

        exp_name = json.dumps(exp_meta, sort_keys=True).encode()
        if exp_name in exp_result_aggregation:
            exp_result_aggregation[exp_name]["result"] += exp_result
            exp_result_aggregation[exp_name]["num"] += 1
        else:
            exp_result_aggregation[exp_name] = {"result" : exp_result, "num" : 1, "meta" : exp_meta}

    if not all_exp_result:
        print("There aren't any experiment results.")
        return

    headers = meta_header + list(metric_header)

    write_csv(exp_dir / "all_exp_result.csv", headers, all_exp_result)

    headers.remove("repeat")

    rows = []
    for val in exp_result_aggregation.values():
        exp_result = val["result"] / val["num"]
        each_exp_result = copy(val["meta"])
        each_exp_result.update(exp_result.get_formatted_result())
        rows.append(each_exp_result)
    write_csv(exp_dir / "exp_summary.csv", headers, rows)

    draw_rich_table(headers, rows, "Experiment Summary")


def perv_aggregate_all_exp_result(exp_dir: Union[str, Path]):
    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)

    tensorboard_dir = exp_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    exp_result_summary = {}
    all_exp_result = []
    header = [
        "val_score",
        "test_score",
        "train_e2e_time",
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
        for key, val in each_exp_result_summary["result"].items():
            if val is None:
                continue
            if key in stdev_field:
                each_exp_result_summary["result"][key] = sqrt(val / each_exp_result_summary["num"])
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


def set_arguments_to_cmd(command: List[str], keys: Union[str, List[str]], value: str = None, start_idx: int = 0):
    """Add arguments at proper position in `sys.argv`.

    Args:
        keys (str or List[str]): arguement keys.
        value (str or None): argument value.
        after_params (bool): whether argument should be after `param` or not.
    """
    if not isinstance(keys, list):
        keys = [keys]
    for key in keys:
        if key in command:
            if value is not None:
                command[command.index(key) + 1] = value
            return

    delimiters = [val.split("_")[1] for val in __all__] + ["params"]

    key = keys[0]
    for i in range(start_idx, len(command)):
        if command[i] in delimiters:
            if value is not None:
                command.insert(i, value)
            command.insert(i, key)
            return


def run_experiment_recipe(exp_recipe: Dict):
    output_path = Path(exp_recipe.get("output_path", f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    output_path.mkdir(exist_ok=True)
    repeat = exp_recipe.get("repeat", 1)

    command_list = get_command_list(exp_recipe)

    current_dir = os.getcwd()
    os.chdir(output_path)
    for command_info in command_list:
        original_command = command_info["command"]
        command_var = command_info["variable"]
        exp_name = "_".join(command_var.values())

        for repeat_idx in range(repeat):
            workspace = Path(exp_name.replace('/', '_') + f"_repeat_{repeat_idx}")
            command_var["repeat"] = str(repeat_idx)

            command = copy(original_command).split()
            if command[1] in ["train", "run"]:
                set_arguments_to_cmd(command, "--workspace", str(workspace), 2)
                if "train" in command:
                    set_arguments_to_cmd(command, "--seed", str(repeat_idx), command.index("train")+1)

            sys.argv = [" ".join(command[:2])] + command[2:]
            globals()["_".join(sys.argv[0].split())]()

            if command[1] in ["train", "run"]:
                organize_exp_result(workspace, command_var)
    os.chdir(current_dir)

    aggregate_all_exp_result(output_path)


def main():
    exp_recipe = get_exp_recipe()
    run_experiment_recipe(exp_recipe)

    return dict(retcode=0)


if __name__ == "__main__":
    main()
