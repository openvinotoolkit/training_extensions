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
from pathlib import Path
from itertools import product

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

    for val in variable.values():
        if isinstance(val, list):
            for idx in range(len(val)):
                pat_ret = replace_pat.search(val[idx])
                if pat_ret is not None:
                    pat_word = pat_ret.groups()[0]
                    val[idx] = val[idx].replace(f"${{{pat_word}}}", variable[pat_word])

    command_arr = []
    pat_ret = set(replace_pat.findall(command))
    if pat_ret:
        temp = []
        for val in pat_ret:
            if isinstance(variable[val], list):
                temp.append(variable[val])
            else:
                temp.append([variable[val]])
        comb = list(product(*temp))
        for each in comb:
            new_commnad = command
            for key, val in zip(pat_ret, each):
                new_commnad = new_commnad.replace(f"${{{key}}}", val)
            command_arr.append(new_commnad)
    else:
        command_arr = [command]

    for command in command_arr:
        sys.argv = [" ".join(command.split()[:2])] + command.split()[2:]
        for _ in range(repeat):
            current_dir = os.getcwd()
            os.chdir(output_path)
            globals()["_".join(sys.argv[0].split())]()
            os.chdir(current_dir)

    return dict(retcode=0)


if __name__ == "__main__":
    main()
