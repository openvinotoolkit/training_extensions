# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from logging import Logger

import pytest
import torch.nn as nn
from mmcv.runner import IterLoader
from torch.utils.data.dataloader import DataLoader

from otx.algorithms.common.adapters.mmcv.runner import (
    EpochRunnerWithCancel,
    IterBasedRunnerWithCancel,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
    load_test_dataset,
)


class TestRunnersInputParamsValidation:
    def iter_based_runner(self):
        return IterBasedRunnerWithCancel(model=self.MockModel(), logger=Logger(name="test logger"))

    @staticmethod
    def data_loader():
        dataset = load_test_dataset()[0]
        return DataLoader(dataset)

    class MockModel(nn.Module):
        @staticmethod
        def train_step():
            pass

    @e2e_pytest_unit
    def test_epoch_runner_with_cancel_train_params_validation(self):
        """
        <b>Description:</b>
        Check EpochRunnerWithCancel object "train" method input parameters validation

        <b>Input data:</b>
        EpochRunnerWithCancel object. "data_loader" non DataLoader object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "train" method
        """
        runner = EpochRunnerWithCancel(model=self.MockModel(), logger=Logger(name="test logger"))
        with pytest.raises(ValueError):
            runner.train(data_loader="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_iter_based_runner_with_cancel_main_loop_params_validation(self):
        """
        <b>Description:</b>
        Check IterBasedRunnerWithCancel object "main_loop" method input parameters validation

        <b>Input data:</b>
        IterBasedRunnerWithCancel object. "main_loop" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "main_loop" method
        """
        data_loader = self.data_loader()
        iter_loader = IterLoader(data_loader)
        runner = self.iter_based_runner()
        correct_values_dict = {
            "workflow": [("train", 1)],
            "iter_loaders": [iter_loader],
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "workflow" parameter
            ("workflow", unexpected_int),
            # Unexpected integer is specified as nested workflow
            ("workflow", [("train", 1), unexpected_int]),
            # Unexpected integer is specified as "iter_loaders" parameter
            ("iter_loaders", unexpected_int),
            # Unexpected integer is specified as nested iter_loader
            ("iter_loaders", [iter_loader, unexpected_int]),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=runner.main_loop,
        )

    @e2e_pytest_unit
    def test_iter_based_runner_with_cancel_run_params_validation(self):
        """
        <b>Description:</b>
        Check IterBasedRunnerWithCancel object "run" method input parameters validation

        <b>Input data:</b>
        IterBasedRunnerWithCancel object. "run" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "run" method
        """
        data_loader = self.data_loader()
        runner = self.iter_based_runner()
        correct_values_dict = {
            "data_loaders": [data_loader],
            "workflow": [("train", 1)],
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "data_loaders" parameter
            ("data_loaders", unexpected_int),
            # Unexpected integer is specified as nested data_loader
            ("data_loaders", [data_loader, unexpected_int]),
            # Unexpected integer is specified as "workflow" parameter
            ("workflow", unexpected_int),
            # Unexpected integer is specified as nested workflow
            ("workflow", [("train", 1), unexpected_int]),
            # Unexpected string is specified as "max_iters" parameter
            ("max_iters", "unexpected string"),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=runner.run,
        )
