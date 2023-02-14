import pytest
from tempfile import TemporaryDirectory


@pytest.fixture
def tmp_dir():
    with TemporaryDirectory() as tmp_dir:
        yield tmp_dir