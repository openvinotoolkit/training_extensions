import pytest

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component

from ote_cli.tools import ote


class TestTelemetry():

    @e2e_pytest_component
    def test_tm_integration(self):
        ote.demo()
