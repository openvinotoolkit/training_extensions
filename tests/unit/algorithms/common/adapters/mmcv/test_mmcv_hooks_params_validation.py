# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from logging import Logger

import pytest
import torch.nn as nn
from mmcv.runner import EpochBasedRunner

from otx.algorithms.common.adapters.mmcv.hooks import (
    CancelTrainingHook,
    EarlyStoppingHook,
    EnsureCorrectBestCheckpointHook,
    OTXLoggerHook,
    OTXProgressHook,
    ReduceLROnPlateauLrUpdaterHook,
    StopLossNanTrainingHook,
)
from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)

# TODO: Need to add EMAMomentumUpdateHook unit-test


class TestCancelTrainingHook:
    @e2e_pytest_unit
    def test_cancel_training_hook_initialization_params_validation(self):
        """
        <b>Description:</b>
        Check CancelTrainingHook object initialization parameters validation

        <b>Input data:</b>
        "interval" non-int type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        CancelTrainingHook object initialization parameter
        """
        with pytest.raises(ValueError):
            CancelTrainingHook(interval="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_cancel_training_hook_after_train_iter_params_validation(self):
        """
        <b>Description:</b>
        Check CancelTrainingHook object "after_train_iter" method input parameters validation

        <b>Input data:</b>
        CancelTrainingHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_train_iter" method
        """
        hook = CancelTrainingHook()
        with pytest.raises(ValueError):
            hook.after_train_iter(runner="unexpected string")  # type: ignore


class TestEnsureCorrectBestCheckpointHook:
    @e2e_pytest_unit
    def test_ensure_correct_best_checkpoint_hook_after_run_params_validation(self):
        """
        <b>Description:</b>
        Check EnsureCorrectBestCheckpointHook object "after_run" method input parameters validation

        <b>Input data:</b>
        EnsureCorrectBestCheckpointHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_run" method
        """
        hook = EnsureCorrectBestCheckpointHook()
        with pytest.raises(ValueError):
            hook.after_run(runner="unexpected string")  # type: ignore


class TestOTXLoggerHook:
    @e2e_pytest_unit
    def test_otx_logger_hook_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check OTXLoggerHook object initialization parameters validation

        <b>Input data:</b>
        OTXLoggerHook object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTXLoggerHook object initialization parameter
        """
        correct_values_dict = {}
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "curves" parameter
            ("curves", unexpected_str),
            # Unexpected string is specified as nested curve
            (
                "curves",
                {
                    "expected": OTXLoggerHook.Curve(),
                    "unexpected": unexpected_str,
                },
            ),
            # Unexpected string is specified as "interval" parameter
            ("interval", unexpected_str),
            # Unexpected string is specified as "ignore_last" parameter
            ("ignore_last", unexpected_str),
            # Unexpected string is specified as "reset_flag" parameter
            ("reset_flag", unexpected_str),
            # Unexpected string is specified as "by_epoch" parameter
            ("by_epoch", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OTXLoggerHook,
        )

    @e2e_pytest_unit
    def test_otx_logger_hook_log_params_validation(self):
        """
        <b>Description:</b>
        Check OTXLoggerHook object "log" method input parameters validation

        <b>Input data:</b>
        OTXLoggerHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "log" method
        """
        hook = OTXLoggerHook()
        with pytest.raises(ValueError):
            hook.log(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_otx_logger_hook_after_train_epoch_params_validation(self):
        """
        <b>Description:</b>
        Check OTXLoggerHook object "after_train_epoch" method input parameters validation

        <b>Input data:</b>
        OTXLoggerHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_train_epoch" method
        """
        hook = OTXLoggerHook()
        with pytest.raises(ValueError):
            hook.after_train_epoch(runner="unexpected string")  # type: ignore


class TestOTXProgressHook:
    @staticmethod
    def time_monitor():
        return TimeMonitorCallback(num_epoch=10, num_train_steps=5, num_test_steps=5, num_val_steps=4)

    def hook(self):
        return OTXProgressHook(time_monitor=self.time_monitor())

    @e2e_pytest_unit
    def test_otx_progress_hook_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check OTXProgressHook object initialization parameters validation

        <b>Input data:</b>
        OTXProgressHook object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTXProgressHook object initialization parameter
        """
        correct_values_dict = {"time_monitor": self.time_monitor()}
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "time_monitor" parameter
            ("time_monitor", unexpected_str),
            # Unexpected string is specified as "verbose" parameter
            ("verbose", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OTXProgressHook,
        )

    @e2e_pytest_unit
    def test_otx_progress_hook_before_run_params_validation(self):
        """
        <b>Description:</b>
        Check OTXProgressHook object "before_run" method input parameters validation

        <b>Input data:</b>
        OTXProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "before_run" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.before_run(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_otx_progress_hook_before_epoch_params_validation(self):
        """
        <b>Description:</b>
        Check OTXProgressHook object "before_epoch" method input parameters validation

        <b>Input data:</b>
        OTXProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "before_epoch" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.before_epoch(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_otx_progress_hook_after_epoch_params_validation(self):
        """
        <b>Description:</b>
        Check OTXProgressHook object "after_epoch" method input parameters validation

        <b>Input data:</b>
        OTXProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_epoch" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.after_epoch(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_otx_progress_hook_before_iter_params_validation(self):
        """
        <b>Description:</b>
        Check OTXProgressHook object "before_iter" method input parameters validation

        <b>Input data:</b>
        OTXProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "before_iter" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.before_iter(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_otx_progress_hook_after_iter_params_validation(self):
        """
        <b>Description:</b>
        Check OTXProgressHook object "after_iter" method input parameters validation

        <b>Input data:</b>
        OTXProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_iter" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.after_iter(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_otx_progress_hook_before_val_iter_params_validation(self):
        """
        <b>Description:</b>
        Check OTXProgressHook object "before_val_iter" method input parameters validation

        <b>Input data:</b>
        OTXProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "before_val_iter" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.before_val_iter(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_otx_progress_hook_after_val_iter_params_validation(self):
        """
        <b>Description:</b>
        Check OTXProgressHook object "after_val_iter" method input parameters validation

        <b>Input data:</b>
        OTXProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_val_iter" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.after_val_iter(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_otx_progress_hook_after_run_params_validation(self):
        """
        <b>Description:</b>
        Check OTXProgressHook object "after_run" method input parameters validation

        <b>Input data:</b>
        OTXProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_run" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.after_run(runner="unexpected string")  # type: ignore


class TestEarlyStoppingHook:
    @staticmethod
    def hook():
        return EarlyStoppingHook(interval=5)

    @e2e_pytest_unit
    def test_early_stopping_hook_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check EarlyStoppingHook object initialization parameters validation

        <b>Input data:</b>
        EarlyStoppingHook object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        EarlyStoppingHook object initialization parameter
        """
        correct_values_dict = {"interval": 5}
        unexpected_dict = {"unexpected": "dictionary"}
        unexpected_values = [
            # Unexpected dictionary is specified as "interval" parameter
            ("interval", unexpected_dict),
            # Unexpected dictionary is specified as "metric" parameter
            ("metric", unexpected_dict),
            # Unexpected dictionary is specified as "rule" parameter
            ("rule", unexpected_dict),
            # Unexpected dictionary is specified as "patience" parameter
            ("patience", unexpected_dict),
            # Unexpected dictionary is specified as "iteration_patience" parameter
            ("iteration_patience", unexpected_dict),
            # Unexpected dictionary is specified as "min_delta" parameter
            ("min_delta", unexpected_dict),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=EarlyStoppingHook,
        )

    @e2e_pytest_unit
    def test_early_stopping_hook_before_run_params_validation(self):
        """
        <b>Description:</b>
        Check EarlyStoppingHook object "before_run" method input parameters validation

        <b>Input data:</b>
        EarlyStoppingHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "before_run" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.before_run(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_early_stopping_hook_after_train_iter_params_validation(self):
        """
        <b>Description:</b>
        Check EarlyStoppingHook object "after_train_iter" method input parameters validation

        <b>Input data:</b>
        EarlyStoppingHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_train_iter" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.after_train_iter(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_early_stopping_hook_after_train_epoch_params_validation(self):
        """
        <b>Description:</b>
        Check EarlyStoppingHook object "after_train_epoch" method input parameters validation

        <b>Input data:</b>
        EarlyStoppingHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_train_epoch" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.after_train_epoch(runner="unexpected string")  # type: ignore


class TestReduceLROnPlateauLrUpdaterHook:
    @staticmethod
    def hook():
        return ReduceLROnPlateauLrUpdaterHook(min_lr=0.1, interval=5)

    class MockModel(nn.Module):
        @staticmethod
        def train_step():
            pass

    @e2e_pytest_unit
    def test_reduce_lr_hook_initialization_parameters_validation(self):
        """
        <b>Description:</b>
        Check ReduceLROnPlateauLrUpdaterHook object initialization parameters validation

        <b>Input data:</b>
        ReduceLROnPlateauLrUpdaterHook object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        ReduceLROnPlateauLrUpdaterHook object initialization parameter
        """
        correct_values_dict = {"min_lr": 0.1, "interval": 5}
        unexpected_dict = {"unexpected": "dictionary"}
        unexpected_values = [
            # Unexpected dictionary is specified as "min_lr" parameter
            ("min_lr", unexpected_dict),
            # Unexpected dictionary is specified as "interval" parameter
            ("interval", unexpected_dict),
            # Unexpected dictionary is specified as "metric" parameter
            ("metric", unexpected_dict),
            # Unexpected dictionary is specified as "rule" parameter
            ("rule", unexpected_dict),
            # Unexpected dictionary is specified as "factor" parameter
            ("factor", unexpected_dict),
            # Unexpected dictionary is specified as "patience" parameter
            ("patience", unexpected_dict),
            # Unexpected dictionary is specified as "iteration_patience" parameter
            ("iteration_patience", unexpected_dict),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=ReduceLROnPlateauLrUpdaterHook,
        )

    @e2e_pytest_unit
    def test_reduce_lr_hook_get_lr_params_validation(self):
        """
        <b>Description:</b>
        Check ReduceLROnPlateauLrUpdaterHook object "get_lr" method input parameters validation

        <b>Input data:</b>
        ReduceLROnPlateauLrUpdaterHook object, "get_lr" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_lr" method
        """
        hook = self.hook()
        runner = EpochBasedRunner(model=self.MockModel(), logger=Logger(name="test logger"))
        correct_values_dict = {"runner": runner, "base_lr": 0.2}
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "runner" parameter
            ("runner", unexpected_str),
            # Unexpected string is specified as "base_lr" parameter
            ("base_lr", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=hook.get_lr,
        )

    @e2e_pytest_unit
    def test_reduce_lr_hook_before_run_params_validation(self):
        """
        <b>Description:</b>
        Check ReduceLROnPlateauLrUpdaterHook object "before_run" method input parameters validation

        <b>Input data:</b>
        ReduceLROnPlateauLrUpdaterHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "before_run" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.before_run(runner="unexpected string")  # type: ignore


class TestStopLossNanTrainingHook:
    @e2e_pytest_unit
    def test_stop_loss_nan_train_hook_after_train_iter_params_validation(self):
        """
        <b>Description:</b>
        Check StopLossNanTrainingHook object "after_train_iter" method input parameters validation

        <b>Input data:</b>
        StopLossNanTrainingHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_train_iter" method
        """
        hook = StopLossNanTrainingHook()
        with pytest.raises(ValueError):
            hook.after_train_iter(runner="unexpected string")  # type: ignore
