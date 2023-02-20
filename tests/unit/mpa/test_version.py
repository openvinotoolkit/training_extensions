from otx.mpa.version import __version__, get_version
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_version():
    return_value = get_version()

    assert return_value == __version__
