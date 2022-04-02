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

import os
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
    "--input": "data/airport/train",
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


def get_pretrained_artifacts(template, root, ote_dir):
    _, template_work_dir, _ = get_some_vars(template, root)
    pretrained_artifact_path = f"{template_work_dir}/trained_{template.model_template_id}"
    logger.debug(f">>> Current pre-trained artifact: {pretrained_artifact_path}")
    if not os.path.exists(pretrained_artifact_path):
        command_args = [
            template.model_template_id,
            "--train-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--train-ann-file"])}',
            "--train-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--train-data-roots"])}',
            "--val-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--val-ann-file"])}',
            "--val-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--val-data-roots"])}',
            "--save-model-to",
            pretrained_artifact_path]
        ote_common(template, root, 'train', command_args)
        assert os.path.exists(pretrained_artifact_path), f"The folder must exists after command execution"
        weights = os.path.join(pretrained_artifact_path, 'weights.pth')
        labels = os.path.join(pretrained_artifact_path, 'label_schema.json')
        assert os.path.exists(weights), f"The {weights} must exists after command execution"
        assert os.path.exists(labels), f"The {labels} must exists after command execution"


def get_exported_artifact(template, root):
    _, template_work_dir, _ = get_some_vars(template, root)
    pretrained_weights_path = os.path.join(f"{template_work_dir}/trained_{template.model_template_id}", "weights.pth")
    assert os.path.exists(pretrained_weights_path), f"The weights must be available by path {pretrained_weights_path}"
    exported_artifact_path = f"{template_work_dir}/exported_{template.model_template_id}"
    logger.debug(f">>> Current exported artifact: {exported_artifact_path}")
    if not os.path.exists(exported_artifact_path):
        command_args = [
            template.model_template_id,
            "--load-weights",
            pretrained_weights_path,
            "--save-model-to",
            exported_artifact_path]
        ote_common(template, root, 'export', command_args)
        openvino_xml = os.path.join(exported_artifact_path, "openvino.xml")
        assert os.path.exists(openvino_xml), f"openvino.xml must exists after export"
        openvino_bin = os.path.join(exported_artifact_path, "openvino.bin")
        assert os.path.exists(openvino_bin), f"openvino.bin must exists after export"
        label_schema_json = os.path.join(exported_artifact_path, "label_schema.json")
        assert os.path.exists(label_schema_json), f"label_schema.json must exists after export"
