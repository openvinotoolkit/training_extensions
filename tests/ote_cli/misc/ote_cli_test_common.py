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


from subprocess import run  # nosec
import logging

from ote_cli.utils.tests import get_some_vars, collect_env_vars

default_train_args_paths = {
    "--train-ann-file": "data/airport/annotation_example_train.json",
    "--train-data-roots": "data/airport/train",
    "--val-ann-file": "data/airport/annotation_example_train.json",
    "--val-data-roots": "data/airport/train",
    "--test-ann-files": "data/airport/annotation_example_train.json",
    "--test-data-roots": "data/airport/train",
}

wrong_paths = {
    "empty": "",
    "not_printable": "\x11",
    # "null_symbol": "\x00" It is caught on subprocess level
}

logger = logging.getLogger(__name__)


def ote_common(template, root, tool, cmd_args):
    work_dir, __, _ = get_some_vars(template, root)
    command_line = ["ote", tool, *cmd_args]
    ret = run(command_line, env=collect_env_vars(work_dir), capture_output=True)
    output = {
        "exit_code": int(ret.returncode),
        "stdout": str(ret.stdout),
        "stderr": str(ret.stderr),
    }
    logger.debug(f"Command arguments: {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {output['stdout']}\n")
    logger.debug(f"Stderr: {output['stderr']}\n")
    logger.debug(f"Exit_code: {output['exit_code']}\n")
    return output
