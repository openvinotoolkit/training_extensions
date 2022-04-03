"""Tests for input parameters with OTE CLI eval tool"""
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

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component

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
    parser_templates,
    root,
    ote_dir,
    eval_args,

)

params_values, params_ids, params_values_for_be, params_ids_for_be = parser_templates()


class TestEvalCommon:
    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values)
    def create_venv_fx(self, template):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        create_venv(algo_backend_dir, work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_eval_no_template(self, back_end, template, create_venv_fx):
        error_string = "the following arguments are required: template"
        command_args = []
        ret = ote_common(template, root, "eval", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_eval_no_test_files(self, back_end, template, create_venv_fx):
        error_string = (
            "ote eval: error: the following arguments are required: --test-ann-files"
        )
        command_args = [
            template.model_template_id,
            "--test-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--test-data-roots"])}',
            "--load-weights",
            f"./trained_{template.model_template_id}/weights.pth",
            "--save-performance",
            f"./trained_{template.model_template_id}/performance.json",
        ]
        ret = ote_common(template, root, "eval", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_eval_no_test_roots(self, back_end, template, create_venv_fx):
        error_string = (
            "ote eval: error: the following arguments are required: --test-data-roots"
        )
        command_args = [
            template.model_template_id,
            "--test-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--test-ann-files"])}',
            "--load-weights",
            f"./trained_{template.model_template_id}/weights.pth",
            "--save-performance",
            f"./trained_{template.model_template_id}/performance.json",
        ]
        ret = ote_common(template, root, "eval", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_eval_no_weights(self, back_end, template, create_venv_fx):
        error_string = (
            "ote eval: error: the following arguments are required: --load-weights"
        )
        command_args = [
            template.model_template_id,
            "--test-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--test-ann-files"])}',
            "--test-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--test-data-roots"])}',
            "--save-performance",
            f"./trained_{template.model_template_id}/performance.json",
        ]
        ret = ote_common(template, root, "eval", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_eval_wrong_paths_test_ann_file(
        self, back_end, template, create_venv_fx
    ):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_args = [
                template.model_template_id,
                "--test-ann-file",
                case,
                "--test-data-roots",
                f'{os.path.join(ote_dir, default_train_args_paths["--test-data-roots"])}',
                "--load-weights",
                f"./trained_{template.model_template_id}/weights.pth",
                "--save-performance",
                f"./trained_{template.model_template_id}/performance.json",
            ]
            ret = ote_common(template, root, "eval", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_eval_wrong_paths_test_data_roots(
        self, back_end, template, create_venv_fx
    ):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_args = [
                template.model_template_id,
                "--test-ann-file",
                f'{os.path.join(ote_dir, default_train_args_paths["--test-ann-files"])}',
                "--test-data-roots",
                case,
                "--load-weights",
                f"./trained_{template.model_template_id}/weights.pth",
                "--save-performance",
                f"./trained_{template.model_template_id}/performance.json",
            ]
            ret = ote_common(template, root, "eval", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_eval_wrong_paths_load_weights(
        self, back_end, template, create_venv_fx
    ):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_args = [
                template.model_template_id,
                "--test-ann-file",
                f'{os.path.join(ote_dir, default_train_args_paths["--test-ann-files"])}',
                "--test-data-roots",
                f'{os.path.join(ote_dir, default_train_args_paths["--test-data-roots"])}',
                "--load-weights",
                case,
                "--save-performance",
                f"./trained_{template.model_template_id}/performance.json",
            ]
            ret = ote_common(template, root, "eval", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"
            assert (
                error_string in ret["stderr"]
            ), f"Different error message {ret['stderr']}"


class TestEvalDetectionTemplateArguments:
    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def create_venv_fx(self, template):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        create_venv(algo_backend_dir, work_dir)

    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def get_pretrained_artifacts_fx(self, template, create_venv_fx):
        get_pretrained_artifacts(template, root, ote_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_eval_pp_confidence_threshold_type(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "invalid float value"
        test_params = [
            "params",
            "--postprocessing.confidence_threshold",
            "String",
        ]
        command_args = eval_args(template, default_train_args_paths, ote_dir, root, additional=test_params)
        ret = ote_common(template, root, "eval", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_eval_pp_confidence_threshold(
        self, template, get_pretrained_artifacts_fx, create_venv_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        pre_trained_weights = (
            f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        )
        logger.debug(f"Pre-trained weights path: {pre_trained_weights}")
        assert os.path.exists(
            pre_trained_weights
        ), f"Pre trained weights must be before test starts"
        test_params = [
            "params",
            "--postprocessing.confidence_threshold",
            "0.5",
        ]
        command_args = eval_args(template, default_train_args_paths, ote_dir, root, additional=test_params)
        ret = ote_common(template, root, "eval", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_eval_pp_confidence_threshold_oob(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "is out of bounds"
        oob_values = ["-0.1", "1.1"]
        for value in oob_values:
            test_params = [
                "params",
                "--postprocessing.confidence_threshold",
                value,
            ]
            command_args = eval_args(template, default_train_args_paths, ote_dir, root, additional=test_params)
            ret = ote_common(template, root, "eval", command_args)
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
    def test_ote_eval_pp_result_based_confidence_threshold_type(
        self, template, create_venv_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        error_string = "Boolean value expected"
        test_params = [
            "params",
            "--postprocessing.result_based_confidence_threshold",
            "NonBoolean",
        ]
        command_args = eval_args(template, default_train_args_paths, ote_dir, root, additional=test_params)
        ret = ote_common(template, root, "eval", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
        assert error_string in ret["stderr"], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize(
        "template",
        params_values_for_be["DETECTION"],
        ids=params_ids_for_be["DETECTION"],
    )
    def test_ote_eval_pp_result_based_confidence_threshold(
        self, template, get_pretrained_artifacts_fx, create_venv_fx
    ):
        _, template_work_dir, _ = get_some_vars(template, root)
        pre_trained_weights = (
            f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        )
        logger.debug(f"Pre-trained weights path: {pre_trained_weights}")
        assert os.path.exists(
            pre_trained_weights
        ), f"Pre trained weights must be before test starts"
        test_params = [
            "params",
            "--postprocessing.result_based_confidence_threshold",
            "False",
        ]
        command_args = eval_args(template, default_train_args_paths, ote_dir, root, additional=test_params)
        ret = ote_common(template, root, "eval", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"
