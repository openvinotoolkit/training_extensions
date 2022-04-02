"""Tests for input parameters with OTE CLI train tool"""
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
import pytest
from copy import deepcopy

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component
from ote_cli.registry import Registry

from ote_cli.utils.tests import (
    create_venv,
    get_some_vars,
)

from ote_cli_test_common import (
    default_train_args_paths,
    wrong_paths,
    ote_common,
    logger,
    get_pretrained_artifacts,
    get_exported_artifact,
)


root = "/tmp/ote_cli/"
ote_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
external_path = os.path.join(ote_dir, "external")


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

COMMON_ARGS = [
    "--train-ann-file",
    f'{os.path.join(ote_dir, default_train_args_paths["--train-ann-file"])}',
    "--train-data-roots",
    f'{os.path.join(ote_dir, default_train_args_paths["--train-data-roots"])}',
    "--val-ann-file",
    f'{os.path.join(ote_dir, default_train_args_paths["--val-ann-file"])}',
    "--val-data-roots",
    f'{os.path.join(ote_dir, default_train_args_paths["--val-data-roots"])}',
    "--load-weights",
]


class TestOptimizeCommon:
    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values)
    def create_venv_fx(self, template):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        create_venv(algo_backend_dir, work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_template(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: template"
        command_line = []
        ret = ote_common(template, root, "optimize", command_line)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_train_ann_file(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: --train-ann-files"
        command_line = [
            template.model_template_id,
            "--train-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--train-data-roots"])}',
            "--val-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--val-ann-file"])}',
            "--val-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--val-data-roots"])}',
            "--load-weights",
            f"./exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"./trained_{template.model_template_id}",
        ]
        ret = ote_common(template, root, "optimize", command_line)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_train_data_roots(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: --train-data-roots"
        command_line = [
            template.model_template_id,
            "--train-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--train-ann-file"])}',
            "--val-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--val-ann-file"])}',
            "--val-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--val-data-roots"])}',
            "--load-weights",
            f"./exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"./trained_{template.model_template_id}",
        ]
        ret = ote_common(template, root, "optimize", command_line)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_val_ann_file(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: --val-ann-files"
        command_line = [
            template.model_template_id,
            "--train-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--train-ann-file"])}',
            "--train-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--train-data-roots"])}',
            "--val-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--val-data-roots"])}',
            "--load-weights",
            f"./exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"./trained_{template.model_template_id}",
        ]
        ret = ote_common(template, root, "optimize", command_line)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_val_data_roots(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: --val-data-roots"
        command_line = [
            template.model_template_id,
            "--train-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--train-ann-file"])}',
            "--train-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--train-data-roots"])}',
            "--val-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--val-ann-file"])}',
            "--load-weights",
            f"./exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"./trained_{template.model_template_id}",
        ]
        ret = ote_common(template, root, "optimize", command_line)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_save_model_to(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: --save-model-to"
        command_line = [
            template.model_template_id,
            "--train-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--train-ann-file"])}',
            "--train-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--train-data-roots"])}',
            "--val-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--val-ann-file"])}',
            "--val-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--val-data-roots"])}',
            "--load-weights",
            f"./exported_{template.model_template_id}/openvino.xml",
        ]
        ret = ote_common(template, root, "optimize", command_line)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_no_load_weights(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: --load-weights"
        command_line = [
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
            f"./trained_{template.model_template_id}",
        ]
        ret = ote_common(template, root, "optimize", command_line)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_path_train_ann_file(
        self, back_end, template, create_venv_fx
    ):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [
                template.model_template_id,
                "--train-ann-file",
                case,
                "--train-data-roots",
                f'{os.path.join(ote_dir, default_train_args_paths["--train-data-roots"])}',
                "--val-ann-file",
                f'{os.path.join(ote_dir, default_train_args_paths["--val-ann-file"])}',
                "--val-data-roots",
                f'{os.path.join(ote_dir, default_train_args_paths["--val-data-roots"])}',
                "--load-weights",
                f"./exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"./trained_{template.model_template_id}",
            ]
            ret = ote_common(template, root, "optimize", command_line)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_paths_train_data_roots(
        self, back_end, template, create_venv_fx
    ):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [
                template.model_template_id,
                "--train-ann-file",
                f'{os.path.join(ote_dir, default_train_args_paths["--train-ann-file"])}',
                "--train-data-roots",
                case,
                "--val-ann-file",
                f'{os.path.join(ote_dir, default_train_args_paths["--val-ann-file"])}',
                "--val-data-roots",
                f'{os.path.join(ote_dir, default_train_args_paths["--val-data-roots"])}',
                "--load-weights",
                f"./exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"./trained_{template.model_template_id}",
            ]
            ret = ote_common(template, root, "optimize", command_line)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_paths_val_ann_file(
        self, back_end, template, create_venv_fx
    ):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [
                template.model_template_id,
                "--train-ann-file",
                f'{os.path.join(ote_dir, default_train_args_paths["--train-ann-file"])}',
                "--train-data-roots",
                f'{os.path.join(ote_dir, default_train_args_paths["--train-data-roots"])}',
                "--val-ann-file",
                case,
                "--val-data-roots",
                f'{os.path.join(ote_dir, default_train_args_paths["--val-data-roots"])}',
                "--load-weights",
                f"./exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"./trained_{template.model_template_id}",
            ]
            ret = ote_common(template, root, "optimize", command_line)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_paths_val_data_roots(
        self, back_end, template, create_venv_fx
    ):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [
                template.model_template_id,
                "--train-ann-file",
                f'{os.path.join(ote_dir, default_train_args_paths["--train-ann-file"])}',
                "--train-data-roots",
                f'{os.path.join(ote_dir, default_train_args_paths["--train-data-roots"])}',
                "--val-ann-file",
                f'{os.path.join(ote_dir, default_train_args_paths["--val-ann-file"])}',
                "--val-data-roots",
                case,
                "--load-weights",
                f"./exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"./trained_{template.model_template_id}",
            ]
            ret = ote_common(template, root, "optimize", command_line)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_saved_model_to(
        self, back_end, template, create_venv_fx
    ):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [
                template.model_template_id,
                *COMMON_ARGS,
                f"./exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                case,
            ]
            ret = ote_common(template, root, "optimize", command_line)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_load_weights(self, back_end, template, create_venv_fx):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [
                template.model_template_id,
                *COMMON_ARGS,
                case,
                "--save-model-to",
                f"./trained_{template.model_template_id}",
            ]
            ret = ote_common(template, root, "optimize", command_line)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_optimize_wrong_save_performance(
        self, back_end, template, create_venv_fx
    ):
        for case in wrong_paths.values():
            command_line = [
                template.model_template_id,
                *COMMON_ARGS,
                f"./exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"./trained_{template.model_template_id}",
                "--save-performance",
                case,
            ]
            ret = ote_common(template, root, "optimize", command_line)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"


class TestOptimizeDetectionTemplateArguments:
    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def create_venv_fx(self, template):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def get_pretrained_artifacts_fx(self, template, create_venv_fx):
        get_pretrained_artifacts(template, root, ote_dir)

    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def get_exported_artifacts_fx(
        self, template, create_venv_fx, get_pretrained_artifacts_fx
    ):
        get_exported_artifact(template, root)

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_lp_batch_size_type(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "invalid int value"
        cases = ["1.0", "Alpha"]
        for case in cases:
            command_args = [
                template.model_template_id,
                *COMMON_ARGS,
                f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"{template_work_dir}/trained_{template.model_template_id}",
                "params",
                "--learning_parameters.batch_size",
                case,
            ]
            ret = ote_common(template, root, "optimize", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_lp_batch_size(
        self, template, create_venv_fx, get_exported_artifacts_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        command_args = [
            template.model_template_id,
            *COMMON_ARGS,
            f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"{template_work_dir}/trained_{template.model_template_id}",
            "params",
            "--learning_parameters.num_iters",
            "1",
            "--learning_parameters.batch_size",
            "1",
        ]
        ret = ote_common(template, root, "optimize", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_lp_batch_size_oob(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "is out of bounds."
        cases = ["0", "513"]
        for case in cases:
            command_args = [
                template.model_template_id,
                *COMMON_ARGS,
                f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"{template_work_dir}/trained_{template.model_template_id}",
                "params",
                "--learning_parameters.batch_size",
                case,
            ]
            ret = ote_common(template, root, "optimize", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_lp_learning_rate_type(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "invalid float value"
        command_args = [
            template.model_template_id,
            *COMMON_ARGS,
            f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"{template_work_dir}/trained_{template.model_template_id}",
            "params",
            "--learning_parameters.learning_rate",
            "NotFloat",
        ]
        ret = ote_common(template, root, "optimize", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_lp_learning_rate(
        self, template, create_venv_fx, get_exported_artifacts_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        command_args = [
            template.model_template_id,
            *COMMON_ARGS,
            f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"{template_work_dir}/trained_{template.model_template_id}",
            "params",
            "--learning_parameters.num_iters",
            "1",
            "--learning_parameters.batch_size",
            "1",
            "--learning_parameters.learning_rate",
            "0.01",
        ]
        ret = ote_common(template, root, "optimize", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_lp_learning_rate_oob(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "is out of bounds."
        cases = ["0.0", "0.2"]
        for case in cases:
            command_args = [
                template.model_template_id,
                *COMMON_ARGS,
                f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"{template_work_dir}/trained_{template.model_template_id}",
                "params",
                "--learning_parameters.learning_rate",
                case,
            ]
            ret = ote_common(template, root, "optimize", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_lp_lr_warmup_iters_type(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "invalid int value"
        cases = ["1.0", "Alpha"]
        for case in cases:
            command_args = [
                template.model_template_id,
                *COMMON_ARGS,
                f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"{template_work_dir}/trained_{template.model_template_id}",
                "params",
                "--learning_parameters.learning_rate_warmup_iters",
                case,
            ]
            ret = ote_common(template, root, "optimize", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_lp_lr_warmup_iters(
        self, template, create_venv_fx, get_exported_artifacts_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        command_args = [
            template.model_template_id,
            *COMMON_ARGS,
            f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"{template_work_dir}/trained_{template.model_template_id}",
            "params",
            "--learning_parameters.num_iters",
            "1",
            "--learning_parameters.batch_size",
            "1",
            "--learning_parameters.learning_rate_warmup_iters",
            "1",
        ]
        ret = ote_common(template, root, "optimize", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_lp_lr_warmup_iters_oob(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "is out of bounds."
        oob_values = ["-1", "10001"]
        for value in oob_values:
            command_args = [
                template.model_template_id,
                *COMMON_ARGS,
                f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"{template_work_dir}/trained_{template.model_template_id}",
                "params",
                "--learning_parameters.learning_rate_warmup_iters",
                value,
            ]
            ret = ote_common(template, root, "optimize", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_lp_num_iters_type(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "invalid int value"
        cases = ["1.0", "Alpha"]
        for case in cases:
            command_args = [
                template.model_template_id,
                *COMMON_ARGS,
                f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"{template_work_dir}/trained_{template.model_template_id}",
                "params",
                "--learning_parameters.num_iters",
                case,
            ]
            ret = ote_common(template, root, "optimize", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_lp_num_iters_positive_case(
        self, template, create_venv_fx, get_exported_artifacts_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        command_args = [
            template.model_template_id,
            *COMMON_ARGS,
            f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"{template_work_dir}/trained_{template.model_template_id}",
            "params",
            "--learning_parameters.num_iters",
            "1",
            "--learning_parameters.batch_size",
            "1",
            "--learning_parameters.num_iters",
            "1",
        ]
        ret = ote_common(template, root, "optimize", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_lp_num_iters_oob(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "is out of bounds."
        oob_values = ["0", "1000001"]
        for value in oob_values:
            command_args = [
                template.model_template_id,
                *COMMON_ARGS,
                f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"{template_work_dir}/trained_{template.model_template_id}",
                "params",
                "--learning_parameters.num_iters",
                value,
            ]
            ret = ote_common(template, root, "optimize", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_pp_confidence_threshold_type(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "invalid float value"
        command_args = [
            template.model_template_id,
            *COMMON_ARGS,
            f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"{template_work_dir}/trained_{template.model_template_id}",
            "params",
            "--postprocessing.confidence_threshold",
            "Alpha",
        ]
        ret = ote_common(template, root, "optimize", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_pp_confidence_threshold(
        self, template, create_venv_fx, get_exported_artifacts_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        command_args = [
            template.model_template_id,
            *COMMON_ARGS,
            f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"{template_work_dir}/trained_{template.model_template_id}",
            "params",
            "--learning_parameters.num_iters",
            "1",
            "--learning_parameters.batch_size",
            "1",
            "--postprocessing.confidence_threshold",
            "0.5",
        ]
        ret = ote_common(template, root, "optimize", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_pp_confidence_threshold_oob(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "is out of bounds."
        oob_values = ["-0.9", "1.1"]
        for value in oob_values:
            command_args = [
                template.model_template_id,
                *COMMON_ARGS,
                f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
                "--save-model-to",
                f"{template_work_dir}/trained_{template.model_template_id}",
                "params",
                "--postprocessing.confidence_threshold",
                value,
            ]
            ret = ote_common(template, root, "optimize", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_pp_result_based_confidence_threshold_type(
        self, template, create_venv_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "Boolean value expected"
        command_args = [
            template.model_template_id,
            *COMMON_ARGS,
            f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"{template_work_dir}/trained_{template.model_template_id}",
            "params",
            "--postprocessing.result_based_confidence_threshold",
            "NonBoolean",
        ]
        ret = ote_common(template, root, "optimize", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_optimize_pp_result_based_confidence_threshold(
        self, template, create_venv_fx, get_exported_artifacts_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        command_args = [
            template.model_template_id,
            *COMMON_ARGS,
            f"{template_work_dir}/exported_{template.model_template_id}/openvino.xml",
            "--save-model-to",
            f"{template_work_dir}/trained_{template.model_template_id}",
            "params",
            "--learning_parameters.num_iters",
            "1",
            "--learning_parameters.batch_size",
            "1",
            "--postprocessing.result_based_confidence_threshold",
            "False",
        ]
        ret = ote_common(template, root, "optimize", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"
