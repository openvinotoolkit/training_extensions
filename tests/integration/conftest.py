import pytest
import os

from pathlib import Path
from tempfile import TemporaryDirectory

@pytest.fixture(scope="session")
def tmp_dir_path():
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture(autouse=True)
def set_default_tmp_path(tmp_dir_path):
    origin_tmp_dir = os.environ.get("TMPDIR", None)
    os.environ["TMPDIR"] = str(tmp_dir_path)
    yield
    if origin_tmp_dir is None:
        del os.environ["TMPDIR"]
    else:
        os.environ["TMPDIR"] = origin_tmp_dir
