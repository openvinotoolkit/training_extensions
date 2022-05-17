import pytest
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)

from torchreid_tasks.monitors import MetricsMonitor, DefaultMetricsMonitor


class TestMetricsMonitorParamsValidation:
    @e2e_pytest_unit
    def test_metrics_monitor_init_params_validation(self):
        """
        <b>Description:</b>
        Check MetricsMonitor object initialization parameters validation

        <b>Input data:</b>
        "log_dir" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        MetricsMonitor object initialization parameter
        """
        for unexpected_value in [
            # Unexpected integer is specified as "path" parameter
            1,
            # Empty string is specified as "path" parameter
            "",
            # Path Null character is specified in "path" parameter
            "./null\0char",
            # Path with non-printable character is specified as "path" parameter
            "./non\nprintable",
        ]:
            with pytest.raises(ValueError):
                MetricsMonitor(log_dir=unexpected_value)

    @e2e_pytest_unit
    def test_metrics_monitor_add_scalar_params_validation(self):
        """
        <b>Description:</b>
        Check MetricsMonitor object "add_scalar" method input parameters validation

        <b>Input data:</b>
        MetricsMonitor object. "add_scalar" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "add_scalar" method
        """
        monitor = MetricsMonitor("./")
        correct_values_dict = {
            "capture": "some capture",
            "value": 0.1,
            "timestamp": 1,
        }
        unexpected_dict = {"unexpected": "dictionary"}
        unexpected_values = [
            # Unexpected dictionary is specified as "capture" parameter
            ("capture", unexpected_dict),
            # Unexpected dictionary is specified as "value" parameter
            ("value", unexpected_dict),
            # Unexpected dictionary is specified as "timestamp" parameter
            ("timestamp", unexpected_dict),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=monitor.add_scalar,
        )


class TestDefaultMetricsMonitorParamsValidation:
    @e2e_pytest_unit
    def test_default_metrics_monitor_add_scalar_params_validation(self):
        """
        <b>Description:</b>
        Check DefaultMetricsMonitor object "add_scalar" method input parameters validation

        <b>Input data:</b>
        DefaultMetricsMonitor object, "add_scalar" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "add_scalar" method
        """
        monitor = DefaultMetricsMonitor()
        correct_values_dict = {
            "capture": "some capture",
            "value": 0.1,
            "timestamp": 1,
        }
        unexpected_dict = {"unexpected": "dictionary"}
        unexpected_values = [
            # Unexpected dictionary is specified as "capture" parameter
            ("capture", unexpected_dict),
            # Unexpected dictionary is specified as "value" parameter
            ("value", unexpected_dict),
            # Unexpected dictionary is specified as "timestamp" parameter
            ("timestamp", unexpected_dict),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=monitor.add_scalar,
        )

    @e2e_pytest_unit
    def test_default_metrics_monitor_get_metric_values_params_validation(self):
        """
        <b>Description:</b>
        Check DefaultMetricsMonitor object "get_metric_values" method input parameters validation

        <b>Input data:</b>
        DefaultMetricsMonitor object, "capture" unexpected-type value

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_metric_values" method
        """
        monitor = DefaultMetricsMonitor()
        with pytest.raises(ValueError):
            monitor.get_metric_values(capture=1)  # type: ignore

    @e2e_pytest_unit
    def test_default_metrics_monitor_get_metric_timestamps_params_validation(self):
        """
        <b>Description:</b>
        Check DefaultMetricsMonitor object "get_metric_timestamps" method input parameters
        validation

        <b>Input data:</b>
        DefaultMetricsMonitor object, "capture" unexpected-type value

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_metric_timestamps" method
        """
        monitor = DefaultMetricsMonitor()
        with pytest.raises(ValueError):
            monitor.get_metric_timestamps(capture=1)  # type: ignore
