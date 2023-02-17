from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def tmp_dir():
    with TemporaryDirectory() as tmp_dir:
        yield tmp_dir
