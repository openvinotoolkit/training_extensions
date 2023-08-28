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
from pathlib import Path
from itertools import product
from typing import Union

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

def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", "--file", type=str)

    return parser.parse_known_args()


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

    output_file = (exp_dir / "exp_table.txt").open("w")
    output_file.write("name\t")
    write_type = False

    for each_exp in exp_dir.iterdir():
        exp_result = each_exp / "exp_result.txt"
        if exp_result.exists():
            with exp_result.open("r") as f:
                lines = f.readlines()
                if not write_type:
                    for line in lines:
                        output_file.write(line.split()[0] + '\t')
                    output_file.write('\n')

                output_file.write(each_exp.name + '\t')
                for line in lines:
                    output_file.write(line.split()[1] + '\t')
                output_file.write('\n')


def main():
    exp_file = parse_args()[0].file
    with open(exp_file, "r") as f:
        exp_recipe = yaml.safe_load(f)

    output_path = Path(exp_recipe["output_path"])
    output_path.mkdir(exist_ok=True)
    variable = exp_recipe["variable"]
    command = exp_recipe["command"]
    repeat = exp_recipe["repeat"]

    replace_pat = re.compile(r"\$\{(\S+)\}")

    # for val in variable.values():
    #     if isinstance(val, list):
    #         for idx in range(len(val)):
    #             pat_ret = replace_pat.search(val[idx])
    #             if pat_ret is not None:
    #                 pat_word = pat_ret.groups()[0]
    #                 val[idx] = val[idx].replace(f"${{{pat_word}}}", variable[pat_word])

    # command_arr = []
    # pat_ret = set(replace_pat.findall(command))
    # if pat_ret:
    #     temp = []
    #     for val in pat_ret:
    #         if isinstance(variable[val], list):
    #             temp.append(variable[val])
    #         else:
    #             temp.append([variable[val]])
    #     comb = list(product(*temp))
    #     for each in comb:
    #         new_command = command
    #         exp_name = ""
    #         for key, val in zip(pat_ret, each):
    #             new_command = new_command.replace(f"${{{key}}}", val)

    #             if exp_name:
    #                 exp_name += "_" + val.replace('/', '_')
    #             else:
    #                 exp_name += val.replace('/', '_')

    #         new_command_split = new_command.split()
    #         new_command_split.insert(2, f"--workspace {exp_name}")
    #         new_command = " ".join(new_command_split)
    #         command_arr.append(new_command)
    # else:
    #     command_arr = [command]

    # current_dir = os.getcwd()
    # os.chdir(output_path)
    # for command in command_arr:
    #     sys.argv = [" ".join(command.split()[:2])] + command.split()[2:]
    #     for _ in range(repeat):
    #         globals()["_".join(sys.argv[0].split())]()
    # os.chdir(current_dir)

    # for exp_dir in output_path.iterdir():
    #     organize_exp_result(exp_dir)

    aggregate_exp_result(output_path)

    return dict(retcode=0)


if __name__ == "__main__":
    main()
