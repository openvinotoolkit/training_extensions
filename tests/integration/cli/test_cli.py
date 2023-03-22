"""Tests for OTX CLI commands"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import sys
from unittest.mock import patch

import pytest

from otx.cli.tools import cli
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    otx_build_auto_config,
    otx_build_backbone_testing,
    otx_build_testing,
    otx_find_testing,
    otx_train_auto_config,
)

otx_dir = os.getcwd()


build_backbone_args = [
    ("CLASSIFICATION", "torchvision.mobilenet_v3_large"),
    ("CLASSIFICATION", "mmcls.MMOVBackbone"),
    ("DETECTION", "torchvision.mobilenet_v3_large"),
    ("INSTANCE_SEGMENTATION", "torchvision.mobilenet_v3_large"),
    ("SEGMENTATION", "torchvision.mobilenet_v3_large"),
]
build_backbone_args_ids = [f"{task}_{backbone}" for task, backbone in build_backbone_args]

rebuild_args = {
    "classification": {
        "default": "EfficientNet-B0",
        "--task": "classification",
        "--model": "MobileNet-V3-large-1x",
        "--train-type": "Semisupervised",
    },
    "detection": {"default": "ATSS", "--task": "detection", "--model": "SSD", "--train-type": "Semisupervised"},
}


class TestToolsOTXCLI:
    @e2e_pytest_component
    def test_otx_find(self):
        otx_find_testing()

    @e2e_pytest_component
    @pytest.mark.parametrize("build_backbone_args", build_backbone_args, ids=build_backbone_args_ids)
    def test_otx_backbone_build(self, tmp_dir_path, build_backbone_args):
        tmp_dir_path = tmp_dir_path / build_backbone_args[0] / build_backbone_args[1]
        otx_build_backbone_testing(tmp_dir_path, build_backbone_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("case", rebuild_args.keys())
    def test_otx_build_rebuild(self, tmp_dir_path, case):
        tmp_dir_path = tmp_dir_path / "test_rebuild" / case
        # 1. Only Task
        build_arg = {"--task": rebuild_args[case]["--task"]}
        expected = {"model": rebuild_args[case]["default"], "train_type": "Incremental"}
        otx_build_testing(tmp_dir_path, build_arg, expected=expected)
        # 2. Change Model
        build_arg = {"--model": rebuild_args[case]["--model"]}
        expected = {"model": rebuild_args[case]["--model"], "train_type": "Incremental"}
        otx_build_testing(tmp_dir_path, build_arg, expected=expected)
        # 3. Change Train-type
        build_arg = {"--train-type": rebuild_args[case]["--train-type"]}
        expected = {"model": rebuild_args[case]["--model"], "train_type": rebuild_args[case]["--train-type"]}
        otx_build_testing(tmp_dir_path, build_arg, expected=expected)
        # 4. Change to Default
        build_arg = {"--model": rebuild_args[case]["default"], "--train-type": "Incremental"}
        expected = {"model": rebuild_args[case]["default"], "train_type": "Incremental"}
        otx_build_testing(tmp_dir_path, build_arg, expected=expected)


build_auto_config_args = {
    "classification": {"--train-data-roots": "tests/assets/imagenet_dataset"},
    "classification_with_task": {"--task": "classification", "--train-data-roots": "tests/assets/imagenet_dataset"},
    "detection": {"--train-data-roots": "tests/assets/car_tree_bug"},
    "detection_with_task": {"--task": "detection", "--train-data-roots": "tests/assets/car_tree_bug"},
}


class TestToolsOTXBuildAutoConfig:
    @e2e_pytest_component
    @pytest.mark.parametrize("case", build_auto_config_args.keys())
    def test_otx_build_with_autosplit(self, case, tmp_dir_path):
        otx_dir = os.getcwd()
        tmp_dir_path = tmp_dir_path / "test_build_auto_config" / case
        otx_build_auto_config(root=tmp_dir_path, otx_dir=otx_dir, args=build_auto_config_args[case])


train_auto_config_args = {
    "classification": {"--train-data-roots": "tests/assets/imagenet_dataset"},
    "classification_with_template": {
        "template": "otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml",
        "--train-data-roots": "tests/assets/imagenet_dataset",
    },
    "detection": {"--train-data-roots": "tests/assets/car_tree_bug"},
    "detection_with_template": {
        "template": "otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml",
        "--train-data-roots": "tests/assets/car_tree_bug",
    },
}

train_params = [
    "params",
    "--learning_parameters.num_iters",
    "1",
]


class TestToolsOTXTrainAutoConfig:
    @e2e_pytest_component
    @pytest.mark.parametrize("case", train_auto_config_args.keys())
    def test_otx_train(self, case, tmp_dir_path):
        otx_dir = os.getcwd()
        tmp_dir_path = tmp_dir_path / "test_train_auto_config" / case
        train_auto_config_args[case]["train_params"] = train_params
        otx_train_auto_config(root=tmp_dir_path, otx_dir=otx_dir, args=train_auto_config_args[case])


class TestTelemetryIntegration:
    _CMDS = ["demo", "build", "deploy", "eval", "explain", "export", "find", "optimize", "train"]

    @e2e_pytest_component
    @patch("otx.cli.utils.telemetry.init_telemetry_session", return_value=None)
    @patch("otx.cli.utils.telemetry.close_telemetry_session", return_value=None)
    @patch("otx.cli.utils.telemetry.send_version", return_value=None)
    @patch("otx.cli.utils.telemetry.send_cmd_results", return_value=None)
    def test_tm_integration_exit_0(
        self,
        mock_send_cmd,
        mock_send_version,
        mock_close_tm,
        mock_init_tm,
    ):
        backup_argv = sys.argv
        for cmd in self._CMDS:
            sys.argv = ["otx", cmd]
            with patch(f"otx.cli.tools.cli.otx_{cmd}", return_value=None) as mock_cmd:
                ret = cli.main()

            assert ret == 0
            mock_cmd.assert_called_once()
            mock_init_tm.assert_called_once()
            mock_close_tm.assert_called_once()
            mock_send_cmd.assert_called_once_with(None, cmd, {"retcode": 0})
            # reset mock state
            mock_init_tm.reset_mock()
            mock_close_tm.reset_mock()
            mock_send_cmd.reset_mock()
        sys.argv = backup_argv

    @e2e_pytest_component
    @patch("otx.cli.utils.telemetry.init_telemetry_session", return_value=None)
    @patch("otx.cli.utils.telemetry.close_telemetry_session", return_value=None)
    @patch("otx.cli.utils.telemetry.send_version", return_value=None)
    @patch("otx.cli.utils.telemetry.send_cmd_results", return_value=None)
    def test_tm_integration_exit_exception(
        self,
        mock_send_cmd,
        mock_send_version,
        mock_close_tm,
        mock_init_tm,
    ):
        backup_argv = sys.argv
        for cmd in self._CMDS:
            with patch(f"otx.cli.tools.cli.otx_{cmd}", side_effect=Exception()):
                sys.argv = ["otx", cmd]
                with pytest.raises(Exception) as e:
                    cli.main()

            assert e.type == Exception, f"{e}"
            mock_init_tm.assert_called_once()
            mock_close_tm.assert_called_once()
            mock_send_cmd.assert_called_once_with(None, cmd, {"retcode": -1, "exception": repr(Exception())})
            # reset mock state
            mock_init_tm.reset_mock()
            mock_close_tm.reset_mock()
            mock_send_cmd.reset_mock()
        sys.argv = backup_argv
