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
from copy import deepcopy

from ote_cli.utils.tests import get_some_vars, collect_env_vars
from ote_cli.registry import Registry

root = "/tmp/ote_cli/"
ote_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
external_path = os.path.join(ote_dir, "external")

default_train_args_paths = {
    "--train-ann-files": "data/airport/annotation_example_train.json",
    "--train-data-roots": "data/airport/train",
    "--val-ann-files": "data/airport/annotation_example_train.json",
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


def parser_templates():
    params_values = []
    params_ids = []
    params_values_for_be = {}
    params_ids_for_be = {}

    for back_end_ in (
        "DETECTION",
        "CLASSIFICATION",
        "ANOMALY_CLASSIFICATION",
        "SEGMENTATION",
        "ROTATED_DETECTION",
        "INSTANCE_SEGMENTATION",
    ):
        cur_templates = Registry(external_path).filter(task_type=back_end_).templates
        cur_templates_ids = [template.model_template_id for template in cur_templates]
        params_values += [(back_end_, t) for t in cur_templates]
        params_ids += [back_end_ + "," + cur_id for cur_id in cur_templates_ids]
        params_values_for_be[back_end_] = deepcopy(cur_templates)
        params_ids_for_be[back_end_] = deepcopy(cur_templates_ids)
    return params_values, params_ids, params_values_for_be, params_ids_for_be


def eval_args(
    _template_,
    args_paths,
    _ote_dir_,
    _root_,
    test_ann_file=True,
    taf_path=None,
    test_data_roots=True,
    tdr_path=None,
    l_weights=True,
    lw_path=None,
    save_performance=True,
    sp_path=None,
    additional=None,
):
    _, twd, _ = get_some_vars(_template_, _root_)
    ret_eval_args = [_template_.model_template_id]
    if test_ann_file:
        ret_eval_args.append("--test-ann-file")
        if taf_path is not None:
            ret_eval_args.append(taf_path)
        else:
            ret_eval_args.append(
                f'{os.path.join(_ote_dir_, args_paths["--test-ann-files"])}'
            )

    if test_data_roots:
        ret_eval_args.append("--test-data-roots")
        if tdr_path is not None:
            ret_eval_args.append(tdr_path)
        else:
            ret_eval_args.append(
                f'{os.path.join(_ote_dir_, args_paths["--test-data-roots"])}'
            )

    if l_weights:
        ret_eval_args.append("--load-weights")
        if lw_path is not None:
            ret_eval_args.append(lw_path)
        else:
            ret_eval_args.append(
                f"{twd}/trained_{_template_.model_template_id}/weights.pth"
            )

    if save_performance:
        ret_eval_args.append("--save-performance")
        if sp_path is not None:
            ret_eval_args.append(sp_path)
        else:
            ret_eval_args.append(
                f"{twd}/trained_{_template_.model_template_id}/performance.json"
            )

    if additional:
        ret_eval_args += [*additional]
    return ret_eval_args


def train_args(
    _template_,
    args_paths,
    _ote_dir_,
    _root_,
    train_ann_files=True,
    taf_path=None,
    train_data_roots=True,
    tdr_path=None,
    val_ann_file=True,
    vaf_path=None,
    val_data_roots=True,
    vdr_path=None,
    save_model_to=True,
    smt_path=None,
    l_weights=False,
    lw_path=None,
    additional=None,
):
    _, twd, _ = get_some_vars(_template_, _root_)
    ret_eval_args = [_template_.model_template_id]
    if train_ann_files:
        ret_eval_args.append("--train-ann-files")
        if taf_path is not None:
            ret_eval_args.append(taf_path)
        else:
            ret_eval_args.append(
                f'{os.path.join(_ote_dir_, args_paths["--train-ann-files"])}'
            )

    if train_data_roots:
        ret_eval_args.append("--train-data-roots")
        if tdr_path is not None:
            ret_eval_args.append(tdr_path)
        else:
            ret_eval_args.append(
                f'{os.path.join(_ote_dir_, args_paths["--train-data-roots"])}'
            )

    if val_ann_file:
        ret_eval_args.append("--val-ann-files")
        if vaf_path is not None:
            ret_eval_args.append(vaf_path)
        else:
            ret_eval_args.append(
                f'{os.path.join(_ote_dir_, args_paths["--val-ann-files"])}'
            )

    if val_data_roots:
        ret_eval_args.append("--val-data-roots")
        if vdr_path is not None:
            ret_eval_args.append(vdr_path)
        else:
            ret_eval_args.append(
                f'{os.path.join(_ote_dir_, args_paths["--val-data-roots"])}'
            )

    if save_model_to:
        ret_eval_args.append("--save-model-to")
        if smt_path is not None:
            ret_eval_args.append(smt_path)
        else:
            ret_eval_args.append(
                f"{twd}/trained_{_template_.model_template_id}"
            )

    if l_weights:
        ret_eval_args.append("--load-weights")
        if lw_path is not None:
            ret_eval_args.append(lw_path)
        else:
            ret_eval_args.append(
                f"{twd}/trained_{_template_.model_template_id}/weights.pth"
            )

    if additional:
        ret_eval_args += [*additional]
    return ret_eval_args


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
    pretrained_artifact_path = (
        f"{template_work_dir}/trained_{template.model_template_id}"
    )
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
            pretrained_artifact_path,
        ]
        ote_common(template, root, "train", command_args)
        assert os.path.exists(
            pretrained_artifact_path
        ), f"The folder must exists after command execution"
        weights = os.path.join(pretrained_artifact_path, "weights.pth")
        labels = os.path.join(pretrained_artifact_path, "label_schema.json")
        assert os.path.exists(
            weights
        ), f"The {weights} must exists after command execution"
        assert os.path.exists(
            labels
        ), f"The {labels} must exists after command execution"


def get_exported_artifact(template, root):
    _, template_work_dir, _ = get_some_vars(template, root)
    pretrained_weights_path = os.path.join(
        f"{template_work_dir}/trained_{template.model_template_id}", "weights.pth"
    )
    assert os.path.exists(
        pretrained_weights_path
    ), f"The weights must be available by path {pretrained_weights_path}"
    exported_artifact_path = (
        f"{template_work_dir}/exported_{template.model_template_id}"
    )
    logger.debug(f">>> Current exported artifact: {exported_artifact_path}")
    if not os.path.exists(exported_artifact_path):
        command_args = [
            template.model_template_id,
            "--load-weights",
            pretrained_weights_path,
            "--save-model-to",
            exported_artifact_path,
        ]
        ote_common(template, root, "export", command_args)
        openvino_xml = os.path.join(exported_artifact_path, "openvino.xml")
        assert os.path.exists(openvino_xml), f"openvino.xml must exists after export"
        openvino_bin = os.path.join(exported_artifact_path, "openvino.bin")
        assert os.path.exists(openvino_bin), f"openvino.bin must exists after export"
        label_schema_json = os.path.join(exported_artifact_path, "label_schema.json")
        assert os.path.exists(
            label_schema_json
        ), f"label_schema.json must exists after export"
