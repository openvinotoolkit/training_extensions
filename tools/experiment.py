"""OTX experiment helper."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import csv
import dataclasses
import gc
import json
import os
import re
import shutil
import statistics
import sys
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from otx.cli.tools.cli import main as otx_cli
from rich.console import Console
from rich.table import Table

rich_console = Console()


def get_parser() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="Experiment recipe file.")
    parser.add_argument("-p", "--parse", type=str, help="Workspace path to parse.")
    parser.add_argument("-d", "--dryrun", action="store_true", help="Print experiment commands without execution.")
    return parser


def parse_time_delta_fmt(time_str: str, format: str) -> timedelta:
    """Convert datetime to timedelta.

    Args:
        time_str (str): datetime format string.
        format (str): datetime format.

    Returns:
        timedelta: timedelta converted from datetime.
    """
    return datetime.strptime(time_str, format) - datetime(1900, 1, 1)


def find_latest_file(root_dir: Union[Path, str], file_name: str) -> Union[None, Path]:
    """Find a latest file of matched files.

    Args:
        root_dir (Union[Path, str]): Root directory for searching.
        file_name (str): File name to search. It can constain shell style wild card.

    Returns:
        Union[None, Path]: Latest file path. If file can't be found, return None.
    """
    root_dir = Path(root_dir)
    train_record_files = sorted((root_dir).glob(file_name), reverse=True, key=lambda x: x.stat().st_mtime)
    if not train_record_files:
        return None
    return train_record_files[0]


def cvt_number_to_str(target: Dict):
    """Convert int or float in dict to string.

    Args:
        target (Dict): Dictionary object to change int or float to string in.
    """
    result = copy(target)

    for key, val in result.items():
        if isinstance(val, (int, float)):
            result[key] = str(val)
        elif isinstance(val, list):
            for i in range(len(val)):
                if isinstance(val[i], (int, float)):
                    val[i] = str(val[i])

    return result


class EvalResult:
    """Class to save otx eval output.

    Current OTX eval output has different metrics depending on a task.
    To deal with it, this class can save dynamic metric name.
    Each metric can be set or gotten by both dict-like(ins["metric"]) or class-like(ins.metric) way.
    "add" (only with a class having same metrics) and "true devide" are supported.
    """

    def __getitem__(self, key):
        """Support dict-like way to get attribute."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Support dict-like way to set attribute."""
        setattr(self, key, value)

    def __add__(self, obj: "EvalResult"):
        """Add with a class having same metrics."""
        new_obj = deepcopy(self)
        new_obj_metrics = vars(new_obj).keys()

        if new_obj_metrics != vars(obj).keys():
            raise KeyError(
                "Two objects have different metrics. "
                f"Left operand : {','.join(new_obj_metrics)} / Right operand : {','.join(vars(obj).keys())}"
            )

        for attr in new_obj_metrics:
            new_obj[attr] += obj[attr]
        return new_obj

    def __truediv__(self, divisor: Union[int, float]):
        """Divide each metric in the class."""
        new_obj = deepcopy(self)

        for attr in vars(new_obj).keys():
            new_obj[attr] /= divisor

        return new_obj


@dataclass
class ExperimentResult:
    """Dataclass to manage experiment result.

    It serves not only storing values but also various features.
    For example, it can be added by other ExperimentResult instance and also divided by integer.
    It can provide dictionary format result and also can parse a dictionary with same format as itself.
    """

    val_score: Union[float, None] = None
    train_eval_result: Union[EvalResult, None] = None
    train_e2e_time: Union[timedelta, None] = None
    avg_iter_time: Union[float, None] = None
    std_iter_time: Union[float, None] = None
    avg_data_time: Union[float, None] = None
    std_data_time: Union[float, None] = None
    export_eval_result: Union[EvalResult, None] = None
    max_cpu_mem: Union[float, None] = None
    avg_cpu_util: Union[float, None] = None
    max_gpu_mem: Union[float, None] = None
    avg_gpu_util: Union[float, None] = None
    optimize_eval_result: Union[EvalResult, None] = None
    epoch: Union[int, None] = None

    def get_formatted_result(self) -> Dict:
        """Return dictionary format result."""
        result = dataclasses.asdict(self)
        formatted_result = {}

        for key, val in result.items():
            if val is None:
                continue
            elif key in ["max_cpu_mem", "max_gpu_mem"]:
                formatted_result[f"{key}(GiB)"] = round(val, 2)
            elif key in ["avg_cpu_util", "avg_gpu_util"]:
                formatted_result[f"{key}(%)"] = round(val, 2)
            elif key == "train_e2e_time":
                formatted_result[key] = str(self.train_e2e_time).split(".")[0]
            elif isinstance(val, EvalResult):
                task = key.split('_')[0]
                for metric, score in vars(val).items():
                    formatted_result[f"{metric}({task})"] = round(score, 4)
            elif isinstance(val, float):
                formatted_result[key] = round(val, 4)

        return formatted_result

    def __add__(self, obj: "ExperimentResult"):
        """Add with same class. If None exists, it's skipped."""
        new_obj = deepcopy(self)

        for attr in dataclasses.fields(self):
            self._add_if_not_none(new_obj, obj, attr.name)

        return new_obj

    @staticmethod
    def _add_if_not_none(dst_obj: "ExperimentResult", src_obj: "ExperimentResult", attr: str):
        dst_obj_val = getattr(dst_obj, attr)
        src_obj_val = getattr(src_obj, attr)
        if dst_obj_val is not None and src_obj_val is not None:
            setattr(dst_obj, attr, dst_obj_val + src_obj_val)
        else:
            setattr(dst_obj, attr, None)

    def __truediv__(self, divisor: Union[int, float]):
        """Divide with same class. If None exists, it's skipped."""
        new_obj = deepcopy(self)

        for attr in dataclasses.fields(self):
            self._divide_if_not_none(new_obj, attr.name, divisor)

        return new_obj

    @staticmethod
    def _divide_if_not_none(obj: "ExperimentResult", attr: str, divisor: Union[int, float]):
        obj_val = getattr(obj, attr)
        if obj_val is not None:
            setattr(obj, attr, obj_val / divisor)

    def parse_formatted_dict(self, formatted_dict: Dict):
        """Parse a dictionary with same format."""
        max_mem_pat = re.compile(r"max_.*_mem")
        cpu_util_pat = re.compile(r"avg.*_util")
        eval_result_pat = re.compile(r"(.*)\((.*)\)")

        for key, val in formatted_dict.items():
            max_mem_name = max_mem_pat.search(key)
            cpu_util_name = cpu_util_pat.search(key)
            eval_result_name = eval_result_pat.search(key)

            if max_mem_name is not None:
                max_mem_name = max_mem_name.group(0)
                setattr(self, max_mem_name, val)
            elif cpu_util_name is not None:
                cpu_util_name = cpu_util_name.group(0)
                setattr(self, cpu_util_name, val)
            elif eval_result_name is not None:
                metric = eval_result_name.group(1)
                task = eval_result_name.group(2)
                eval_result = getattr(self, f"{task}_eval_result")
                if  eval_result is None:
                    eval_result = EvalResult()
                    eval_result[metric] = val
                    setattr(self, f"{task}_eval_result", eval_result)
                else:
                    eval_result[metric] = val
            elif key == "train_e2e_time":
                setattr(self, key, parse_time_delta_fmt(val, "%H:%M:%S"))
            else:
                setattr(self, key, val)


class BaseExpParser(ABC):
    """Base class for an experiment parser.

    Args:
        workspace (Path): Workspace to parse.
    """

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._exp_result = ExperimentResult()
        self._iter_time_arr = []
        self._data_time_arr = []

    @abstractmethod
    def parse_exp_log(self):
        """Abstract method to parse experiment log."""
        raise NotImplementedError

    def get_exp_result(self):
        """Get experiment result."""
        self._calculate_avg_std_per_iter()

        return self._exp_result.get_formatted_result()

    def _calculate_avg_std_per_iter(self):
        if self._iter_time_arr:
            self._exp_result.avg_iter_time = statistics.mean(self._iter_time_arr)
            self._exp_result.std_iter_time = (
                statistics.stdev(self._iter_time_arr) if len(self._iter_time_arr) > 1 else 0
            )

        if self._data_time_arr:
            self._exp_result.avg_data_time = statistics.mean(self._data_time_arr)
            self._exp_result.std_data_time = (
                statistics.stdev(self._data_time_arr) if len(self._data_time_arr) > 1 else 0
            )

    def _parse_eval_output(self, file_path: Path):
        for task in ["train", "export", "optimize"]:
            if task in str(file_path.parent.name):
                break
        else:
            print(f"Can not parse eval output in {file_path.parent.name}")
            return

        with file_path.open("r") as f:
            eval_output: Dict = json.load(f)

        eval_result = EvalResult()
        for metric, score in eval_output.items():
            eval_result[metric] = score

        setattr(self._exp_result, f"{task}_eval_result", eval_result)

    def _parse_resource_usage(self, file_path: Path):
        with file_path.open("r") as f:
            resource_usage = yaml.safe_load(f)

        if "cpu" in resource_usage:
            self._exp_result.max_cpu_mem = float(resource_usage["cpu"]["max_memory_usage"].split()[0])
            self._exp_result.avg_cpu_util = float(resource_usage["cpu"]["avg_util"].split()[0])

        if "gpu" in resource_usage:
            self._exp_result.max_gpu_mem = float(resource_usage["gpu"]["total_max_mem"].split()[0])
            self._exp_result.avg_gpu_util = float(resource_usage["gpu"]["total_avg_util"].split()[0])

    def _parse_cli_report(self, file_path: Path, save_val_score=True):
        with file_path.open("r") as f:
            lines = f.readlines()

        val_score_pattern = re.compile(r"score:.*Performance\(score: ([-+]?\d+(\.\d*)?|\.\d+)")
        e2e_time_pattern = re.compile(r"time elapsed: '(\d+:\d+:\d+(\.\d*)?)'")
        for line in lines:
            if save_val_score:
                val_score = val_score_pattern.search(line)
                if val_score is not None:
                    self._exp_result.val_score = float(val_score.group(1))

            e2e_time = e2e_time_pattern.search(line)
            if e2e_time is not None:
                self._exp_result.train_e2e_time = parse_time_delta_fmt(e2e_time.group(1), "%H:%M:%S.%f")


class MMCVExpParser(BaseExpParser):
    """MMCV experiment parser class."""

    def parse_exp_log(self):
        """Parse experiment log."""
        for task_dir in (self._workspace / "outputs").iterdir():
            if task_dir.is_symlink():  # prevent duplicated parse
                continue

            if "train" in str(task_dir.name):
                # test score
                eval_files = list(task_dir.glob("performance.json"))
                if eval_files:
                    self._parse_eval_output(eval_files[0])

                # iter, data time, epoch
                train_record_file = find_latest_file(task_dir / "logs", "*.log.json")
                if train_record_file is not None:
                    self._parse_train_record(train_record_file)

                # train e2e time & val score
                cli_report_files = list(task_dir.glob("cli_report.log"))
                if cli_report_files:
                    self._parse_cli_report(cli_report_files[0])

                # get resource info
                resource_file = task_dir / "resource_usage.yaml"
                if resource_file.exists():
                    self._parse_resource_usage(resource_file)

            elif "export" in str(task_dir) or "optimize" in str(task_dir):
                eval_files = list(task_dir.glob("performance.json"))
                if eval_files:
                    self._parse_eval_output(eval_files[0])

    def _parse_train_record(self, file_path: Path):
        with file_path.open("r") as f:
            lines = f.readlines()

        last_epoch = 0
        for line in lines:
            iter_history = json.loads(line)
            if iter_history.get("mode") == "train":
                self._iter_time_arr.append(iter_history["time"])
                self._data_time_arr.append(iter_history["data_time"])
                if iter_history["epoch"] > last_epoch:
                    last_epoch = iter_history["epoch"]

        self._exp_result.epoch = last_epoch


class AnomalibExpParser(BaseExpParser):
    """Anomalib experiment parser class."""

    def parse_exp_log(self):
        """Parse experiment log."""
        for task_dir in (self._workspace / "outputs").iterdir():
            if task_dir.is_symlink():
                continue

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

            elif "export" in str(task_dir) or "optimize" in str(task_dir):
                eval_files = list(task_dir.glob("performance.json"))
                if eval_files:
                    self._parse_eval_output(eval_files[0])


def get_exp_parser(workspace: Path) -> Union[BaseExpParser, None]:
    """Get experiment parser depending on framework.

    Args:
        workspace (Path): Workspace to parse.

    Returns:
        Union[BaseExpParser, None]: Experiment parser. If template file doesn't exist in the workspace, return None.
    """
    template_file = workspace / "template.yaml"
    if not template_file.exists():
        return None

    with template_file.open("r") as f:
        template = yaml.safe_load(f)

    if "anomaly" in template["task_type"].lower():
        return AnomalibExpParser(workspace)
    return MMCVExpParser(workspace)


def organize_exp_result(workspace: Union[str, Path], exp_meta: Optional[Dict[str, str]] = None):
    """Organize experiment result and save it as a file named exp_result.yaml.

    Args:
        workspace (Union[str, Path]): Workspace to organize an expeirment result.
        exp_meta (Dict[str, str], optional):
            Experiment meta information. If it exists, it's saved together. Defaults to None.
    """
    if isinstance(workspace, str):
        workspace = Path(workspace)

    exp_parser = get_exp_parser(workspace)
    if exp_parser is None:
        print(f"Unable to find which task \"{workspace}\" is. Parsing experiment result is skipped.")
        return
    exp_parser.parse_exp_log()

    exp_result = exp_parser.get_exp_result()

    if not exp_result:
        print(f"There is no experiment result in {workspace}")

    with (workspace / "exp_result.yaml").open("w") as f:
        yaml.dump({"meta": exp_meta, "exp_result": exp_result}, f, default_flow_style=False)


def write_csv(output_path: Union[str, Path], header: List[str], rows: List[Dict[str, Any]]):
    """Write csv file based on header and rows.

    Args:
        output_path (Union[str, Path]): Where file is saved.
        header (List[str]): List of header.
        rows (List[Dict[str, Any]]): Each row of csv.
    """
    if isinstance(output_path, str):
        output_path = Path(output_path)

    with output_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def print_table(headers: List[str], rows: List[Dict[str, Any]], table_title: str = "Table"):
    """Print a table to console.

    Args:
        headers (List[str]): List of headers.
        rows (List[Dict[str, Any]]): Rows of table.
        table_title (str, optional): Table title. Defaults to "Table".
    """
    table = Table(title=table_title)
    for header in headers:
        table.add_column(header, justify="center", no_wrap=True)
    for row in rows:
        table_row = []
        for header in headers:
            val = row.get(header)
            table_row.append(str(val))

        table.add_row(*table_row)

    rich_console.print(table, justify="center", crop=False)


def aggregate_all_exp_result(exp_dir: Union[str, Path]):
    """Aggregate all experiment results and save it and it's summary as a file.

    Args:
        exp_dir (Union[str, Path]): Experiment directory.
    """
    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)

    tensorboard_dir = exp_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    meta_header: Union[List[str], None] = None
    metric_header = set()
    all_exp_result: List[Dict[str, str]] = []
    exp_result_aggregation = {}
    for each_exp in exp_dir.iterdir():
        # parse single experiment
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

        # Sum experiments with same variables.
        exp_name = json.dumps(exp_meta, sort_keys=True).encode()  # get unique hash based on variable
        if exp_name in exp_result_aggregation:
            exp_result_aggregation[exp_name]["result"] += exp_result
            exp_result_aggregation[exp_name]["num"] += 1
        else:
            exp_result_aggregation[exp_name] = {"result": exp_result, "num": 1, "meta": exp_meta}

        # copy tensorboard log into tensorboard dir
        exp_tb_dir = list(each_exp.rglob("tf_logs"))
        if exp_tb_dir:
            shutil.copytree(exp_tb_dir[0], tensorboard_dir / each_exp.name, dirs_exist_ok=True)

    if not all_exp_result:
        print("There aren't any experiment results.")
        return

    # print and save the experiment aggregation
    headers = sorted(meta_header) + sorted(metric_header)
    write_csv(exp_dir / "all_exp_result.csv", headers, all_exp_result)

    for key in ["repeat", "std_iter_time", "std_data_time"]:  # average of std is distorted value
        if key in headers:
            headers.remove(key)

    rows = []
    for val in exp_result_aggregation.values():
        exp_result = val["result"] / val["num"]
        exp_result.std_iter_time = None
        exp_result.std_data_time = None
        each_exp_result = copy(val["meta"])

        each_exp_result.update(exp_result.get_formatted_result())
        rows.append(each_exp_result)
    write_csv(exp_dir / "exp_summary.csv", headers, rows)


@dataclass
class Command:
    """Command dataclass."""

    command: List[str]
    variable: Dict[str, str] = field(default_factory=dict)


class ExpInfo:
    """Class to store experiment information.

    It does additional things to provide complete experiment information.
    For example, it replaces constants or varilabes if necessary,
    and then it makes all possibles commands based on variables.

    Args:
        command (Union[str, List[str]]): All commands to exeucte.
        output_path (Path): Output path to save experiment result.
        name (str, optional): Experiment name. Defaults to "".
        constants (Dict[str, str], optional):
            Constants. If there are constants in variables or commands,
            they are replaced based on this value. Defaults to None.
        variables (Dict[str, str], optional):
            Variables. If there are variables in command, they're replaced based on this value. Defaults to None.
        repeat (int, optional): How many times to repeat experiments. Defaults to 1.
    """

    def __init__(
        self,
        command: Union[str, List[str]],
        output_path: Path,
        name: str = "",
        constants: Optional[Dict[str, str]] = None,
        variables: Optional[Dict[str, str]] = None,
        repeat: int = 1,
    ):
        self._raw_command = command
        self._commands: Optional[List[Command]] = None
        self.output_path = output_path
        self.name = name
        self._constants = constants if constants is not None else {}
        if variables is None:
            variables = {}
        self._raw_variables = cvt_number_to_str(variables)
        self._variables: Optional[Dict[str, str]] = None
        self.repeat = repeat
        self._replace_pat = re.compile(r"\$\{(\w+)\}")

    @property
    def constants(self) -> Dict[str, str]:
        """Constants in recipe file."""
        return self._constants

    @property
    def variables(self) -> Dict[str, Union[str, List[str]]]:
        """Variables in recipe file. If it contains constants, they're replaced by real value."""
        if self._variables is None:
            self._variables = self._replace_var_in_target(self.constants, self._raw_variables)
        return self._variables

    @property
    def commands(self) -> List[Command]:
        """List of commands from experiment recipe.

        It counts all available cases and makes Command instance per each case.

        Returns:
            List[Command]: List of Command instances.
        """
        if self._commands is None:
            command = self._raw_command
            if isinstance(command, str):
                command = [command]
            command = self._replace_var_in_target(self.constants, command)
            var_combinations = self._product_all_cases(self.variables, command)
            if not var_combinations:
                self._commands = [Command(command=command)]
            else:
                command_arr = []
                for var_combination in var_combinations:
                    command_arr.append(Command(self._replace_var_in_target(var_combination, command), var_combination))
                self._commands = command_arr
        return self._commands

    def _product_all_cases(
        self,
        variable: Dict[str, Union[str, List[str]]],
        target_str: Union[str, List[str]],
    ) -> List[Dict[str, str]]:
        if isinstance(target_str, str):
            target_str = [target_str]
        found_keys = set()
        for each_str in target_str:
            found_keys.update([x for x in set(self._replace_pat.findall(each_str)) if x in variable])
        if not found_keys:
            return []

        found_keys = sorted(found_keys)
        values_of_found_key = []
        for key in found_keys:
            if isinstance(variable[key], list):
                values_of_found_key.append(variable[key])
            else:
                values_of_found_key.append([variable[key]])

        all_cases = []
        for value_of_key_found in product(*values_of_found_key):
            all_cases.append(dict(zip(found_keys, value_of_key_found)))

        return all_cases

    def _replace_var_in_target(
        self,
        variable: Dict[str, str],
        target: Union[str, List, Dict],
    ) -> Union[str, List, Dict]:
        if isinstance(target, str):
            for key, val in variable.items():
                target = target.replace(f"${{{key}}}", val)
        elif isinstance(target, list):
            target = target.copy()
            for i in range(len(target)):
                target[i] = self._replace_var_in_target(variable, target[i])
        elif isinstance(target, dict):
            target = target.copy()
            for key in target.keys():
                target[key] = self._replace_var_in_target(variable, target[key])
        else:
            raise TypeError(f"{type(target)} isn't supported type. target should has str, list or dict type.")

        return target



def parse_exp_recipe(recipe_file: Union[str, Path]) -> Tuple[List[ExpInfo], Path]:
    """Parse an experiment recipe and return list of expeirment information and output path.

    Args:
        recipe_file (Union[str, Path]): Recipe file to parse.

    Raises:
        RuntimeError: If recipe file doesn't exist, error is raised.

    Returns:
        Tuple[List[ExpInfo], Path]: List of expeirment information and output path.
    """
    if not os.path.exists(recipe_file):
        raise RuntimeError(f"{recipe_file} doesn't exist.")

    with open(recipe_file, "r") as f:
        exp_recipe: Dict = yaml.safe_load(f)

    ori_constants: Dict[str, str] = cvt_number_to_str(exp_recipe.get("constants", {}))
    output_path = Path(exp_recipe.get("output_path", f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    exp_info_list = []

    if "experiments" not in exp_recipe:
        exp_recipe["experiments"] = [exp_recipe]
        exp_recipe["experiments"][0]["name"] = ""
        exp_recipe["experiments"][0].pop("constants")

    for exp in exp_recipe["experiments"]:
        constants = copy(ori_constants)
        if "constants" in exp:
            constants.update(exp["constants"])

        exp_info_list.append(
            ExpInfo(
                exp["command"],
                output_path / exp["name"],
                exp["name"],
                constants,
                exp.get("variables"),
                exp.get("repeat", 1)
            )
        )

    return exp_info_list, output_path


@dataclass
class CommandFailInfo:
    """Dataclass to store command fail information."""

    exception: Exception
    variable: Dict[str, str]
    command: str

    def get_formatted_result(self) -> Dict:
        """Return dictionary format result."""
        result = dataclasses.asdict(self)
        result["exception"] = str(result["exception"])
        return result


def log_exp_failed_cases(
    failed_cases: Union[List[CommandFailInfo], Dict[str, List[CommandFailInfo]]],
    output_path: Path,
):
    """Print experiments failed cases to console and save them in each experiment directory as a file.

    Args:
        failed_cases (Union[List[CommandFailInfo], Dict[str, List[CommandFailInfo]]]):
            List of CommandFailInfo or Dictionary having experiment name as key and CommandFailInfo object as value.
        output_path (Path): Directory where experiment direcory exists.
    """
    if isinstance(failed_cases, list):
        failed_cases = {"" : failed_cases}

    for exp_name, failed_cases in failed_cases.items():
        rich_console.rule(f"[bold red]{exp_name} failed cases ")

        for each_fail_case in failed_cases:
            rich_console.print(f"Case : {each_fail_case.variable}", crop=False)
            rich_console.print(f"command : {each_fail_case.command}", crop=False)
            rich_console.print("Error log:", str(each_fail_case.exception), crop=False)
            rich_console.print()

        with (output_path / exp_name / "failed_cases.yaml").open("w") as f:
            yaml.safe_dump([fail_case.get_formatted_result() for fail_case in failed_cases], f)


class OtxCommandRunner:
    """Class to run list of otx commands who have same varaibles.

    It provides convenient features not just runs commands.
    Same workspae made at first otx command is used for all commands.
    Therefore, the template don't have to be set after first command.
    And when executing OTX eval, which model to evaluate is decided depending on previous command.

    Args:
        command_ins (Command): Command instance to run.
        repeat_idx (int): repeat index.
    """

    OUTPUT_FILE_NAME: Dict[str, List[str]] = {
        "export": ["openvino.bin"],
        "optimize": ["weights.pth", "openvino.bin"]
    }

    def __init__(self, command_ins: Command, workspace: Path, repeat_idx: int):
        self._command_ins = command_ins
        self._repeat_idx = repeat_idx
        self._command_var = copy(command_ins.variable)
        self._workspace = workspace
        self._command_var["repeat"] = str(repeat_idx)
        self._fail_logs: List[CommandFailInfo] = []
        self._previous_cmd_entry: List[str] = []

    @property
    def fail_logs(self) -> List[CommandFailInfo]:
        """Information of all failed cases."""
        return self._fail_logs

    def run_command_list(self, dryrun: bool = False):
        """Run all commands and organize experiment results."""
        for command in self._command_ins.command:
            command = command.split()
            if not self._prepare_run_command(command) and not dryrun:
                print(f"otx {command[1]} is skipped.")
                continue

            if not dryrun:
                self._run_otx_command(command)
            else:
                print(" ".join(command))

            self._previous_cmd_entry.append(command[1])

            gc.collect()

        if not dryrun:
            organize_exp_result(self._workspace, self._command_var)

    def _prepare_run_command(self, command: List[str]) -> bool:
        self.set_arguments_to_cmd(command, "--workspace", str(self._workspace))
        cmd_entry = command[1]
        previous_cmd = None
        for previous_cmd in reversed(self._previous_cmd_entry):
            if previous_cmd != "eval":
                break

        if cmd_entry == "train":
            self.set_arguments_to_cmd(command, "--seed", str(self._repeat_idx))
        elif cmd_entry == "eval":
            if previous_cmd in ["export", "optimize"]:
                file_path = self._find_model_path(previous_cmd)
                if file_path is None:
                    return False
                self.set_arguments_to_cmd(command, "--load-weights", str(file_path))
                output_path = str(file_path.parents[1])
            else:
                output_path = str(self._workspace / "outputs" / "latest_trained_model")
            self.set_arguments_to_cmd(command, "--output", output_path)
        elif cmd_entry == "optimize":
            if previous_cmd == "export":  # execute PTQ. If not, execute QAT
                file_path = self._find_model_path(previous_cmd)
                if file_path is None:
                    return False
                self.set_arguments_to_cmd(command, "--load-weights", str(file_path))

        return True

    def _run_otx_command(self, command: List[str]):
        sys.argv = copy(command)
        try:
            otx_cli()
        except Exception as e:
            self._fail_logs.append(CommandFailInfo(variable=self._command_var, exception=e, command=" ".join(command)))

    def _find_model_path(self, cmd_entry: str):
        output_dir = find_latest_file(self._workspace / "outputs", f"*{cmd_entry}")
        if output_dir is None:
            print(f"There is no {cmd_entry} output directory.")
            return None
        for file_name in self.OUTPUT_FILE_NAME[cmd_entry]:
            file_path = list(output_dir.rglob(file_name))
            if file_path:
                return file_path[0]

        print(f"{', '.join(self.OUTPUT_FILE_NAME[cmd_entry])} can't be found.")
        return None

    @staticmethod
    def set_arguments_to_cmd(command: List[str], key: str, value: Optional[str] = None, before_params: bool = True):
        """Add arguments at proper position in command.

        Args:
            command (List[str]): list includng a otx command entry and arguments.
            key (str): arguement key.
            value (str or None): argument value.
            before_params (bool): whether argument should be after `param` or not.
        """
        if key in command:
            if value is not None:
                command[command.index(key) + 1] = value
            return

        if before_params and "params" in command:
            index = command.index("params")
        else:
            index = len(command)

        if value is not None:
            command.insert(index, value)
        command.insert(index, key)


def run_experiment(exp_info: ExpInfo, dryrun: bool = False) -> List[CommandFailInfo]:
    """Run single expeirment.

    Args:
        exp_info (ExpInfo): ExpInfo having expreiment information to conduct.
        dryrun (bool, optional): Whether to only print experiment commands. Defaults to False.

    Returns:
        List[CommandFailInfo]: List of failed command information.
    """
    failed_cases: List[CommandFailInfo] = []

    for command_ins in exp_info.commands:
        for repeat_idx in range(exp_info.repeat):
            otx_cmd_runner = OtxCommandRunner(
                command_ins,
                exp_info.output_path
                / Path("_".join(command_ins.variable.values()).replace("/", "_") + f"_repeat_{repeat_idx}"),
                repeat_idx
            )
            otx_cmd_runner.run_command_list(dryrun)
            failed_cases.extend(otx_cmd_runner.fail_logs)

    if not dryrun:
        aggregate_all_exp_result(exp_info.output_path)

    return failed_cases


def print_experiments_summary(output_path: Path):
    """Print experiment summary to console and save it as a file.

    Args:
        output_path (Path): Output path where experiment summary file is saved.
    """
    rich_console.rule("[bold green]Experiment summary")
    
    for summary_file in output_path.rglob("exp_summary.csv"):
        exp_name = summary_file.parent.name
        if not summary_file.exists():
            print(f"{exp_name} doesn't have exp_summary.csv file. Skipped.")
            continue

        with summary_file.open() as f:
            exp_summary_csv = csv.reader(f)

            headers = next(exp_summary_csv)
            rows = []
            for row in exp_summary_csv:
                rows.append(dict((header, val) for header, val in zip(headers, row)))

        print_table(headers, rows, f"{exp_name}")


def run_experiment_recipe(recipe_file: Union[str, Path], dryrun: bool = False):
    """Run experiments based on the recipe.

    Args:
        recipe_file (Union[str, Path]): Recipe file to run.
        dryrun (bool, optional): Whether to only print experiment commands. Defaults to False.
    """
    total_failed_cases: Dict[str, List[CommandFailInfo]] = {}
    exp_info_list, output_path = parse_exp_recipe(recipe_file)
    for exp_info in exp_info_list:
        failed_cases = run_experiment(exp_info, dryrun)
        total_failed_cases[exp_info.name] = failed_cases

    if dryrun:
        return

    for failed_cases in total_failed_cases.values():
        if failed_cases:
            log_exp_failed_cases(total_failed_cases, output_path)
            break

    print_experiments_summary(output_path)


def main():
    """Main function to decide which function to execute."""
    parser = get_parser()
    args = parser.parse_args()

    if args.file is not None and args.parse is not None:
        print("Please give either --file or --parse argument.")
    elif args.file is not None:
        run_experiment_recipe(args.file, args.dryrun)
    elif args.parse is not None:
        organize_exp_result(args.parse)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
