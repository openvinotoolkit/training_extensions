import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--data-dir", type=str, default=None, help="Path to datasets directory for Accuracy Checker"
    )


@pytest.fixture(scope="module")
def data_dir(request):
    return request.config.getoption("--data-dir")
